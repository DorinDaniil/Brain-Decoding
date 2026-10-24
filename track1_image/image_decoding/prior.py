"""Stage 2: a small flow-matching prior that refines the encoder's embedding.

Rectified flow from the encoder's prediction to the true embedding, both in
z-space (see encoder.TargetSpace), conditioned on the prediction:

    x0 = e_hat + sigma * eps,   x1 = z,   x_t = (1 - t) x0 + t x1
    L  = || v(x_t, t, e_hat) - (x1 - x0) ||^2

``e_hat`` is the encoder output put on the sphere of typical targets (the CLIP
loss only fixes its direction). At inference the ODE dx/dt = v(x, t, e_hat)
is integrated with a few Euler steps from x0 = e_hat.

v is a U-Net for vectors: an MLP that narrows (1536 -> 1024 -> 512), widens
back, and concatenates skip connections from the narrowing path. Time and
condition enter every block through FiLM.

Train it on encoder outputs of windows the encoder never saw (val), otherwise
the pairs are too easy: the encoder is overfitted on its own training windows.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_decoding.evaluation import retrieval


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
    def __init__(self, dim=1536, widths=(1024, 512), d_emb=256):
        super().__init__()
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


class FlowPrior:
    def __init__(self, space, dim, sigma=0.1, device="cpu", **unet_kwargs):
        self.space, self.sigma, self.device = space, sigma, device
        self.net = VectorUNet(dim, **unet_kwargs).to(device)

    def _on_sphere(self, e):
        return self.space.radius * F.normalize(e, dim=-1)

    @torch.no_grad()
    def refine(self, e, n_steps=10):
        """Encoder outputs e (N, D) in z-space -> refined z (N, D)."""
        self.net.eval()
        cond = self._on_sphere(e).to(self.device)
        x = cond.clone()
        for i in range(n_steps):
            t = torch.full((len(x),), i / n_steps, device=self.device)
            x = x + self.net(x, t, cond) / n_steps
        return x.cpu()

    def fit(self, e, z, epochs=200, lr=3e-4, batch_size=512, monitor=None, n_steps=10):
        """Pairs (encoder output e, true z). ``monitor = (e, y, subject)`` held out: keep its best top-5."""
        cond, z = self._on_sphere(e).to(self.device), z.to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        best, history = (-1.0, None), []
        for epoch in range(epochs):
            self.net.train()
            for idx in torch.randperm(len(z), device=self.device).split(batch_size):
                c, x1 = cond[idx], z[idx]
                x0 = c + self.sigma * torch.randn_like(c)
                t = torch.rand(len(idx), device=self.device)
                xt = (1 - t[:, None]) * x0 + t[:, None] * x1
                loss = F.mse_loss(self.net(xt, t, c), x1 - x0)
                opt.zero_grad(); loss.backward(); opt.step()
            row = {"epoch": epoch, "loss": loss.item()}
            if monitor is not None:
                e_m, y_m, s_m = monitor
                pred = self.space.to_y(self.refine(e_m, n_steps)).numpy()
                row.update(retrieval(pred, y_m, s_m))
                if row["top5"] > best[0]:
                    best = (row["top5"], copy.deepcopy(self.net.state_dict()))
            history.append(row)
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        return history


class RefinedModel:
    """Track 1 contract with both stages: X -> encoder -> flow refinement -> target space."""

    def __init__(self, encoder_model, prior, n_steps=10):
        self.encoder, self.prior, self.n_steps = encoder_model, prior, n_steps

    def predict(self, X, subject=None):
        z = self.prior.refine(self.encoder.embed(X, subject), self.n_steps)
        return self.prior.space.to_y(z)


@torch.no_grad()
def embed_split(encoder_model, loader):
    """Encoder outputs on a split: (e, y, subject) as tensors / numpy."""
    e, y, s = [], [], []
    for X, target, subject in loader:
        e.append(encoder_model.embed(X)); y.append(target); s.append(subject)
    return torch.cat(e), torch.cat(y).numpy(), torch.cat(s).numpy()


def split_by_image(y, fraction=0.1, seed=0):
    """Boolean mask holding out ``fraction`` of the images (all their windows)."""
    _, idx = np.unique(y, axis=0, return_inverse=True)
    idx = idx.reshape(-1)
    held = np.random.default_rng(seed).permutation(idx.max() + 1)[: max(1, int(fraction * (idx.max() + 1)))]
    return np.isin(idx, held)
