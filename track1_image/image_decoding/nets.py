"""Networks and the inference model, self-contained (torch only).

This file is copied verbatim into the Codabench ``submission.py`` by
``image_decoding.export``, so it must not import anything from this project.

The encoder never sees channel indices, only electrode positions: channels are
merged into virtual channels by attention over Fourier features of their 3D
positions, so the same weights read any montage (32-channel Emotiv, 63-channel
THINGS-EEG2) and missing channels are simply left out. Nothing depends on the
subject: the sealed test is on new people.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_RADIUS = 0.1  # metres: positions are divided by it, so the scalp sits in [-1, 1]


def montage(names, table_names, table):
    """Positions (C, 3) of ``names`` from a name -> xyz table, and which of them were found (C,)."""
    row = {n.upper(): i for i, n in enumerate(table_names)}
    index = torch.tensor([row.get(str(n).upper(), -1) for n in names])
    pos = table[index.clamp(min=0)] / HEAD_RADIUS
    return pos * (index >= 0)[:, None], index >= 0


def fourier_features(pos, n_freqs):
    """(C, 3) positions in [-1, 1] -> (C, 2 n_freqs^3) cosines and sines on an integer frequency grid."""
    grid = torch.arange(n_freqs, device=pos.device, dtype=pos.dtype)
    k = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    angles = math.pi * pos @ k.T
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class ChannelMerger(nn.Module):
    """Real channels -> virtual channels by attention over channel positions (Defossez et al. 2023).

    Virtual channel v mixes the input with softmax_c(q_v . phi(pos_c)) over the
    channels present. During training a random scalp region is dropped from
    each window, so no single electrode is relied on.
    """

    def __init__(self, n_virtual=128, n_freqs=4, drop_prob=0.5, drop_radius=0.35):
        super().__init__()
        self.n_freqs, self.drop_prob, self.drop_radius = n_freqs, drop_prob, drop_radius
        self.queries = nn.Parameter(torch.randn(n_virtual, 2 * n_freqs ** 3) / n_freqs ** 1.5)

    def forward(self, X, pos, valid):
        """X (B, C, T), pos (C, 3), valid (C,) bool -> (B, n_virtual, T)."""
        mask = valid[None].expand(len(X), -1)
        if self.training and self.drop_prob > 0:
            center = pos[torch.multinomial(valid.float().expand(len(X), -1), 1)[:, 0]]  # a present channel
            near = torch.cdist(center[:, None], pos[None].expand(len(X), -1, -1))[:, 0] < self.drop_radius
            dropped = mask & ~(near & (torch.rand(len(X), 1, device=X.device) < self.drop_prob))
            mask = torch.where(dropped.any(1, keepdim=True), dropped, mask)
        scores = self.queries @ fourier_features(pos, self.n_freqs).T  # (V, C)
        weights = scores[None].masked_fill(~mask[:, None], float("-inf")).softmax(-1)
        return torch.einsum("bvc,bct->bvt", weights, X)


class ConvBlock(nn.Module):
    """Residual dilated temporal convolution: x + Dropout(GELU(BN(conv(x))))."""

    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(F.gelu(self.norm(self.conv(x))))


class SpatialConvEncoder(nn.Module):
    """EEG window -> u, the prediction p = mu + s u of the DINOv2 embedding.

    Channel merger -> 1x1 conv + time embedding -> dilated residual convolutions
    -> attention pooling over time (several heads, each its own latency
    profile) -> MLP. p lives in the leaderboard's geometry: it is ranked by
    cosine against the raw DINOv2 embeddings, and its norm weighs it when the
    predictions of repeated presentations are summed.
    """

    def __init__(self, n_times, dim, n_virtual=128, width=320, dilations=(1, 2, 4, 8, 16), n_heads=4,
                 hidden=2048, dropout=0.3, n_freqs=4, drop_prob=0.5, drop_radius=0.35):
        super().__init__()
        self.config = dict(n_times=n_times, dim=dim, n_virtual=n_virtual, width=width, dilations=list(dilations),
                           n_heads=n_heads, hidden=hidden, dropout=dropout, n_freqs=n_freqs,
                           drop_prob=drop_prob, drop_radius=drop_radius)
        self.merger = ChannelMerger(n_virtual, n_freqs, drop_prob, drop_radius)
        self.inp = nn.Sequential(nn.Conv1d(n_virtual, width, 1), nn.BatchNorm1d(width), nn.GELU())
        self.time = nn.Parameter(torch.zeros(1, width, n_times))
        self.blocks = nn.Sequential(*[ConvBlock(width, d, dropout) for d in dilations])
        self.attend = nn.Conv1d(width, n_heads, 1)
        self.head = nn.Sequential(nn.LayerNorm(n_heads * width), nn.Linear(n_heads * width, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, dim))
        nn.init.zeros_(self.head[-1].weight)  # start at p = mu
        nn.init.zeros_(self.head[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(50.0)))  # inverse temperature of the loss

    def forward(self, X, pos, valid):
        h = self.inp(self.merger(X, pos, valid))
        time = self.time if h.shape[-1] == self.time.shape[-1] else F.interpolate(self.time, h.shape[-1])
        h = self.blocks(h + time)
        attention = self.attend(h).softmax(-1)  # (B, heads, T)
        return self.head(torch.einsum("bht,bwt->bhw", attention, h).flatten(1))


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
    """EEG window -> DINOv2-giant embedding: an ensemble of encoders, optionally refined by a flow prior.

    Each member predicts p_m = mu + s u_m; the answer is the mean of p_m / scale_m
    (scale_m: the member's typical norm). Channels are matched by name to a table
    of electrode positions; unknown names are left out. Stateless: every window
    is predicted on its own.
    """

    def __init__(self, encoders, mu, s, radius, positions, ch_names, scales=None, prior=None, n_steps=0):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.prior, self.n_steps = prior, n_steps
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("s", torch.as_tensor(s, dtype=torch.float32))
        self.register_buffer("radius", torch.as_tensor(radius, dtype=torch.float32))
        self.register_buffer("scales", torch.ones(len(encoders)) if scales is None
                             else torch.as_tensor(scales, dtype=torch.float32))
        self.table_names = list(positions)
        self.register_buffer("table", torch.tensor([positions[n] for n in self.table_names], dtype=torch.float32))
        self.ch_names = [str(c) for c in ch_names]
        self.match_channels(self.ch_names)

    def match_channels(self, input_names):
        """Read inputs in the order of ``input_names``; returns how many have a known position."""
        pos, valid = montage(input_names, self.table_names, self.table.cpu())
        if not valid.any():
            raise ValueError(f"no known electrode among {list(input_names)[:10]}")
        self.register_buffer("pos", pos.to(self.mu.device), persistent=False)
        self.register_buffer("valid", valid.to(self.mu.device), persistent=False)
        return int(valid.sum())

    def ensemble(self, X):
        p = [(self.mu + self.s * enc(X, self.pos, self.valid)) / scale for enc, scale in zip(self.encoders, self.scales)]
        return torch.stack(p).mean(0)

    @torch.no_grad()
    def refine(self, p):
        """Euler steps of the flow prior from p put on the sphere of typical targets (in z = (y - mu) / s)."""
        cond = (self.radius * F.normalize(p, dim=-1) - self.mu) / self.s
        x = cond.clone()
        for i in range(self.n_steps):
            t = torch.full((len(x),), i / self.n_steps, device=x.device)
            x = x + self.prior(x, t, cond) / self.n_steps
        return self.mu + self.s * x

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        p = self.ensemble(torch.as_tensor(X, dtype=torch.float32).to(self.mu.device))
        return self.refine(p) if self.prior is not None and self.n_steps > 0 else p
