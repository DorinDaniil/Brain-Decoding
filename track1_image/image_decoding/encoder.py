"""Stage 1: train the encoder in the leaderboard's own geometry.

The leaderboard ranks a prediction p by cos(p, y_j) against the raw DINOv2
embeddings y_j, and in the sealed phase it first sums the predictions of all
repetitions of an image by one person. Training does the same:

- a batch holds G images, each with V *views*; a view is k repetitions of the
  image by one subject, and its prediction is the mean of their predictions
  (k drawn from ``ks`` each step), as in the subject-aggregated score;
  views of one image by different subjects are all positives for it;
- InfoNCE on cos(p, y) / temperature over the batch images, both directions;
  optionally soft targets (a little mass on images similar to the true one)
  and an MSE between p and y in units of the target spread. Both are off by
  default: on synthetic data they lowered held-out top-5 (45% -> 30%).

Validation is a dress rehearsal of the sealed phase: subjects held out of
training entirely, on the study's test images, scored subject-aggregated.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from image_decoding.evaluation import evaluate
from image_decoding.nets import ImageDecoder, SpatialConvEncoder


def target_space(data):
    """mu, s and the typical norm of the training images' embeddings."""
    y = torch.as_tensor(data.images[np.unique(data.image[~data.test])])
    mu = y.mean(0)
    return mu, (y - mu).std(), y.norm(dim=-1).mean()


def held_out_subjects(data, n, seed=0):
    """``n`` subjects with test windows, kept out of training to stand in for the sealed phase's new people."""
    candidates = np.unique(data.subject[data.test])
    return np.sort(np.random.default_rng(seed).choice(candidates, min(n, len(candidates) - 1), replace=False))


def split(data, held_out):
    """Window indices: training (other subjects, training images) and validation (held out, test images)."""
    held = np.isin(data.subject, held_out)
    return np.flatnonzero(~held & ~data.test), np.flatnonzero(held & data.test)


class ViewSampler:
    """Draws G images and, for each, V views of k repetitions by one subject."""

    def __init__(self, image, subject, index, seed=0):
        index = np.asarray(index)
        self.order = index[np.lexsort((subject[index], image[index]))]  # sorted by (image, subject)
        pair = image[self.order].astype(np.int64) * (int(subject.max()) + 1) + subject[self.order]
        _, self.start, self.count = np.unique(pair, return_index=True, return_counts=True)
        self.images, self.pair_start, self.pair_count = np.unique(
            image[self.order[self.start]], return_index=True, return_counts=True)
        self.rng = np.random.default_rng(seed)
        self.queue = np.empty(0, int)

    def batch(self, n_images, views, k):
        """Image ids (G,) and window indices (G, V, k); every image appears once per epoch of the queue."""
        if len(self.queue) < n_images:
            self.queue = self.rng.permutation(len(self.images))
        pick, self.queue = self.queue[:n_images], self.queue[n_images:]
        pairs = self.pair_start[pick, None] + (self.rng.random((n_images, views)) * self.pair_count[pick, None]).astype(int)
        trials = self.start[pairs][..., None] + (self.rng.random((n_images, views, k)) * self.count[pairs][..., None]).astype(int)
        return self.images[pick], self.order[trials]


def contrastive_loss(p, y, label, log_scale, mu, s, soft=0.0, soft_tau=0.1, mse_weight=0.0):
    """p (N, D) view predictions, y (G, D) batch images, label (N,) image of each view."""
    logits = log_scale.clamp(max=math.log(1000)).exp() * F.normalize(p, dim=-1) @ F.normalize(y, dim=-1).T
    hard = F.one_hot(label, len(y)).float()
    target = hard
    if soft > 0:
        c = F.normalize(y - mu, dim=-1)  # image similarity without the common mean
        target = (1 - soft) * hard + soft * (c @ c.T / soft_tau).softmax(-1)[label]
    eeg_to_image = F.cross_entropy(logits, target)
    image_to_eeg = F.cross_entropy(logits.T, hard.T / hard.T.sum(1, keepdim=True))
    loss = (eeg_to_image + image_to_eeg) / 2
    return loss + mse_weight * ((p - y[label]) / s).pow(2).mean() if mse_weight > 0 else loss


def train_encoder(data, held_out, epochs=30, lr=3e-4, weight_decay=0.05, images_per_batch=256, views=2,
                  ks=(1, 2, 4), soft=0.0, mse_weight=0.0, device="cpu", seed=0, log=print, **net_kwargs):
    """Train one encoder; keep the epoch with the best subject-aggregated top-5 on held-out subjects.

    Returns (encoder, history).
    """
    torch.manual_seed(seed)
    mu, s, radius = target_space(data)
    train_idx, val_idx = split(data, held_out)
    net = SpatialConvEncoder(data.X.shape[-1], data.images.shape[1], **net_kwargs)
    model = ImageDecoder([net], mu, s, radius, data.positions, data.ch_names).to(device)
    mu, s = model.mu, model.s
    images = torch.as_tensor(data.images, device=device)
    sampler = ViewSampler(data.image, data.subject, train_idx, seed)
    rng = np.random.default_rng(seed)

    G = min(images_per_batch, len(sampler.images))
    steps = max(1, round(len(train_idx) / (G * views * np.mean(ks))))
    total, warmup = epochs * steps, min(500, epochs * steps // 10)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(i / total, 1.0))))
    amp = torch.device(device).type == "cuda"

    best, history = (-1.0, None), []
    log(f"training on {device}: {len(train_idx)} windows, {len(sampler.images)} images, {steps} steps per epoch; "
        f"val: subjects {held_out.tolist()}, {len(val_idx)} windows of {len(np.unique(data.image[val_idx]))} "
        f"test images (chance top-5 {500 / len(np.unique(data.image[val_idx])):.2f}%)")
    for epoch in range(epochs):
        net.train()
        for step in range(steps):
            k = int(rng.choice(ks))
            image, idx = sampler.batch(G, views, k)
            X = data.windows(idx.reshape(-1), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                u = net(X, model.pos, model.valid)
            p = (mu + s * u.float()).view(G * views, k, -1).mean(1)
            label = torch.arange(G, device=device).repeat_interleave(views)
            loss = contrastive_loss(p, images[torch.as_tensor(image, device=device)], label, net.log_scale,
                                    mu, s, soft, mse_weight=mse_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            schedule.step()
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}/{steps} k={k} loss {loss.item():.3f}")

        scores = evaluate(model, data, val_idx)
        history.append({"epoch": epoch, "loss": loss.item(), **scores})
        log(f"epoch {epoch}: held-out subjects, top-5 subject-agg {scores['top5_subject_agg']:.2f}, "
            f"per window {scores['top5']:.2f}")
        if scores["top5_subject_agg"] > best[0]:
            best = (scores["top5_subject_agg"], copy.deepcopy(net.state_dict()))

    net.load_state_dict(best[1])
    return net.eval(), history
