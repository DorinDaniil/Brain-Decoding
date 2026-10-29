"""Stage 1: train the EEG encoder with a CLIP loss.

Embeddings live in a centered, scaled target space ``z = (y - mu) / s``:
DINOv2 embeddings share a large common mean, which would otherwise dominate
the cosine similarities. The CLIP loss pulls cos(e_i, z_i) up against the
other images of the batch; repeated presentations of one image are all
positives, not false negatives. tau is learned, the loss is symmetric.
"""

import copy

import torch
import torch.nn.functional as F

from image_decoding.evaluation import evaluate
from image_decoding.nets import EEGEncoder, ImageDecoder


def target_space(loader, n_batches=100):
    """mu, s and the typical norm of z = (y - mu) / s, from the training targets."""
    ys = torch.unique(torch.cat([y for (_, y, _), _ in zip(loader, range(n_batches))]), dim=0)
    mu = ys.mean(0)
    s = (ys - mu).std()
    return mu, s, ((ys - mu) / s).norm(dim=-1).mean()


def clip_loss(e, z, log_tau):
    e, z = F.normalize(e, dim=-1), F.normalize(z, dim=-1)
    logits = log_tau.exp() * e @ z.T
    # exact distances: the default matmul trick leaves ~1e-3 on the diagonal
    same = (torch.cdist(z, z, compute_mode="donot_use_mm_for_euclid_dist") < 1e-6).float()
    target = same / same.sum(1, keepdim=True)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def train_encoder(loaders, meta, epochs=30, lr=3e-4, weight_decay=1e-2, device="cpu", log=print, **net_kwargs):
    """Train on ``train``, keep the epoch with the best val top-5. Returns (ImageDecoder, history)."""
    mu, s, radius = target_space(loaders["train"])
    net = EEGEncoder(meta["n_chans"], meta["n_times"], meta["dim"], meta["n_subjects"], **net_kwargs)
    model = ImageDecoder(net, mu, s, radius, meta["ch_names"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    best, history = (-1.0, None), []
    n_batches = len(loaders["train"])
    log(f"training on {device}: {n_batches} batches per epoch")
    for epoch in range(epochs):
        net.train()
        for i, (X, y, subject) in enumerate(loaders["train"]):
            z = ((y - mu) / s).to(device)
            loss = clip_loss(net(X.to(device), subject.to(device)), z, net.log_tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 200 == 0:
                log(f"epoch {epoch} batch {i}/{n_batches} loss {loss.item():.3f}")

        val = evaluate(model, loaders["val"])  # a new person: average subject channel
        history.append({"epoch": epoch, "loss": loss.item(), **val})
        log(f"epoch {epoch}: val top-5 {val['top5']:.2f}, top-1 {val['top1']:.2f}")
        if val["top5"] > best[0]:
            best = (val["top5"], copy.deepcopy(net.state_dict()))

    net.load_state_dict(best[1])
    return model, history
