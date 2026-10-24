"""Stage 1: EEG encoder trained with a CLIP loss.

Embeddings live in a centered, scaled target space ``z = (y - mu) / s``:
DINOv2 embeddings share a large common mean, which would otherwise dominate
the cosine similarities. The encoder maps a window X to e in that space; the
CLIP loss pulls cos(e_i, z_i) up against the other images of the batch:

    L = CE( tau * cos(e, z)^T , P ),   P_ij ∝ [image_i == image_j]

Repeated presentations of one image in a batch are all positives (P), not
false negatives. tau is learned, the loss is symmetric (EEG->image and back).

Subject channel: every training subject has a learnable time series of length
T, appended to its EEG as one more channel, so the spatial filters can use
(and cancel) a subject-specific component. A new person gets the average of
the learned channels; during training a window's own channel is replaced by
that average with probability ``subject_dropout``, so the encoder also works
without knowing who it is. The platform passes only X: there the average
channel is always used.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_decoding.evaluation import retrieval


class TargetSpace:
    """y <-> z = (y - mu) / s, fitted on training targets."""

    def __init__(self, loader, n_batches=100):
        ys = torch.cat([y for (_, y, _), _ in zip(loader, range(n_batches))])
        ys = torch.unique(ys, dim=0)
        self.mu = ys.mean(0)
        self.s = (ys - self.mu).std()
        self.radius = ((ys - self.mu) / self.s).norm(dim=-1).mean()  # typical norm of z

    def to_z(self, y):
        return (y - self.mu) / self.s

    def to_y(self, z):
        """Back to the target space; z is first put on the sphere of typical targets."""
        return self.mu + self.s * self.radius * F.normalize(z, dim=-1)


class EEGEncoder(nn.Module):
    """Subject channel -> temporal conv -> spatial conv over all channels -> projector (NICE-style)."""

    def __init__(self, n_chans, n_times, dim, n_subjects, k=40, dropout=0.5, subject_dropout=0.2):
        super().__init__()
        self.subject_channel = nn.Parameter(0.1 * torch.randn(n_subjects, n_times))
        self.subject_dropout = subject_dropout
        self.convs = nn.Sequential(
            nn.Conv2d(1, k, (1, 13)), nn.AvgPool2d((1, 25), (1, 5)), nn.BatchNorm2d(k), nn.ELU(),
            nn.Conv2d(k, k, (n_chans + 1, 1)), nn.BatchNorm2d(k), nn.ELU(), nn.Dropout(dropout),
            nn.Conv2d(k, k, 1), nn.Flatten(),
        )
        n_flat = self.convs(torch.zeros(2, 1, n_chans + 1, n_times)).shape[1]
        self.project = nn.Linear(n_flat, dim)
        self.residual = nn.Sequential(nn.GELU(), nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
        self.log_tau = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, X, subject=None):
        """X (B, C, T); subject (B,) training-subject ids, or None for a new person."""
        average = self.subject_channel.mean(0).expand(len(X), -1)
        if subject is None:
            extra = average
        else:
            extra = self.subject_channel[subject]
            if self.training:
                drop = torch.rand(len(X), 1, device=X.device) < self.subject_dropout
                extra = torch.where(drop, average, extra)
        X = torch.cat([X, extra[:, None]], dim=1)  # (B, C + 1, T)
        h = self.project(self.convs(X[:, None]))
        return self.norm(h + self.residual(h))


def clip_loss(e, z, log_tau):
    e, z = F.normalize(e, dim=-1), F.normalize(z, dim=-1)
    logits = log_tau.exp() * e @ z.T
    # exact distances: the default matmul trick leaves ~1e-3 on the diagonal
    same = (torch.cdist(z, z, compute_mode="donot_use_mm_for_euclid_dist") < 1e-6).float()
    target = same / same.sum(1, keepdim=True)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


class EncoderModel:
    """Track 1 contract: ``predict(X) -> (B, D)`` in the original target space."""

    def __init__(self, net, space, device):
        self.net, self.space, self.device = net, space, device

    @torch.no_grad()
    def embed(self, X, subject=None):
        """Encoder output e in z-space; ``subject=None``: a new person (average channel)."""
        self.net.eval()
        if subject is not None:
            subject = torch.as_tensor(subject).to(self.device)
        return self.net(torch.as_tensor(X, dtype=torch.float32).to(self.device), subject).cpu()

    def predict(self, X, subject=None):
        return self.space.to_y(self.embed(X, subject))


def train_encoder(loaders, meta, epochs=30, lr=3e-4, weight_decay=1e-2, device="cpu", **net_kwargs):
    """Train on ``train``, keep the epoch with the best val top-5. Returns (model, history)."""
    space = TargetSpace(loaders["train"])
    net = EEGEncoder(meta["n_chans"], meta["n_times"], meta["dim"], meta["n_subjects"], **net_kwargs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    model = EncoderModel(net, space, device)

    best, history = (-1.0, None), []
    for epoch in range(epochs):
        net.train()
        for X, y, subject in loaders["train"]:
            opt.zero_grad()
            loss = clip_loss(net(X.to(device), subject.to(device)), space.to_z(y).to(device), net.log_tau)
            loss.backward()
            opt.step()

        pred, target, subject = [], [], []
        for X, y, s in loaders["val"]:
            pred.append(model.predict(X)); target.append(y); subject.append(s)
        val = retrieval(torch.cat(pred).numpy(), torch.cat(target).numpy(), torch.cat(subject).numpy())
        history.append({"epoch": epoch, "loss": loss.item(), **val})
        if val["top5"] > best[0]:
            best = (val["top5"], copy.deepcopy(net.state_dict()))

    net.load_state_dict(best[1])
    return model, history
