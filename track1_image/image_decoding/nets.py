"""Networks and the inference model, self-contained (torch only).

This file is copied verbatim into the Codabench ``submission.py`` by
``image_decoding.export``, so it must not import anything from this project.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    """Subject channel -> temporal conv -> spatial conv over all channels -> projector.

    Every training subject has a learnable time series appended to its EEG as one
    more channel. A new person gets the average of the learned channels; during
    training a window's own channel is replaced by that average with probability
    ``subject_dropout``, so the encoder also works without knowing who it is.
    """

    def __init__(self, n_chans, n_times, dim, n_subjects, k=40, dropout=0.5, subject_dropout=0.2):
        super().__init__()
        self.config = dict(n_chans=n_chans, n_times=n_times, dim=dim, n_subjects=n_subjects, k=k,
                           dropout=dropout, subject_dropout=subject_dropout)
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
        self.log_tau = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

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


class Block(nn.Module):
    """LayerNorm -> Linear -> FiLM(emb) -> GELU."""

    def __init__(self, d_in, d_out, d_emb):
        super().__init__()
        self.norm, self.lin = nn.LayerNorm(d_in), nn.Linear(d_in, d_out)
        self.film = nn.Linear(d_emb, 2 * d_out)

    def forward(self, h, emb):
        scale, shift = self.film(emb).chunk(2, dim=-1)
        return F.gelu(self.lin(self.norm(h)) * (1 + scale) + shift)


class VectorUNet(nn.Module):
    """Velocity field v(x, t, cond) of the flow prior: an MLP U-Net with FiLM conditioning."""

    def __init__(self, dim=1536, widths=(1024, 512), d_emb=256):
        super().__init__()
        self.config = dict(dim=dim, widths=list(widths), d_emb=d_emb)
        self.d_emb = d_emb
        self.time = nn.Sequential(nn.Linear(d_emb, d_emb), nn.GELU(), nn.Linear(d_emb, d_emb))
        self.cond = nn.Sequential(nn.Linear(dim, d_emb), nn.GELU(), nn.Linear(d_emb, d_emb))
        sizes = [dim, *widths]
        self.down = nn.ModuleList([Block(a, b, d_emb) for a, b in zip(sizes[:-1], sizes[1:])])
        self.mid = Block(sizes[-1], sizes[-1], d_emb)
        self.up = nn.ModuleList([Block(2 * b, a, d_emb) for a, b in zip(sizes[:-1], sizes[1:])][::-1])
        self.out = nn.Linear(2 * dim, dim)

    def _time_features(self, t):
        freqs = torch.exp(-math.log(1000) * torch.arange(self.d_emb // 2, device=t.device) / (self.d_emb // 2))
        angles = t[:, None] * freqs[None] * 1000
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, x, t, cond):
        emb = self.time(self._time_features(t)) + self.cond(cond)
        skips, h = [x], x
        for block in self.down:
            h = block(h, emb)
            skips.append(h)
        h = self.mid(h, emb)
        for block in self.up:
            h = block(torch.cat([h, skips.pop()], dim=-1), emb)
        return self.out(torch.cat([h, skips.pop()], dim=-1))


class ImageDecoder(nn.Module):
    """EEG window -> DINOv2-giant embedding: encoder, optional flow refinement, back to target space.

    Targets live in z = (y - mu) / s; predictions are put on the sphere of
    typical targets (radius) before going back to y. Channels are matched by
    name to the training montage; missing ones are zero. Stateless: every
    window is predicted on its own.
    """

    def __init__(self, encoder, mu, s, radius, ch_names, prior=None, n_steps=10):
        super().__init__()
        self.encoder, self.prior, self.n_steps = encoder, prior, n_steps
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("s", torch.as_tensor(s, dtype=torch.float32))
        self.register_buffer("radius", torch.as_tensor(radius, dtype=torch.float32))
        self.ch_names = [str(c) for c in ch_names]
        self.channel_index = None  # input channel feeding each training channel, -1 = missing

    def match_channels(self, input_names):
        """Map the evaluation montage onto the training one by channel name (case-insensitive)."""
        position = {str(n).upper(): i for i, n in enumerate(input_names)}
        self.channel_index = torch.tensor([position.get(n.upper(), -1) for n in self.ch_names])
        return int((self.channel_index >= 0).sum())

    def _select(self, X):
        if self.channel_index is None:
            return X
        index = self.channel_index.to(X.device)
        out = X[:, index.clamp(min=0)]
        return out * (index >= 0)[None, :, None]

    def on_sphere(self, e):
        return self.radius * F.normalize(e, dim=-1)

    @torch.no_grad()
    def refine(self, e):
        """Euler steps of the flow prior from the encoder output (in z-space)."""
        cond = self.on_sphere(e)
        x = cond.clone()
        for i in range(self.n_steps):
            t = torch.full((len(x),), i / self.n_steps, device=x.device)
            x = x + self.prior(x, t, cond) / self.n_steps
        return x

    @torch.no_grad()
    def predict(self, X, subject=None):
        self.eval()
        X = self._select(torch.as_tensor(X, dtype=torch.float32).to(self.mu.device))
        z = self.encoder(X, subject)
        if self.prior is not None and self.n_steps > 0:
            z = self.refine(z)
        return self.mu + self.s * self.on_sphere(z)
