"""Stage 2: a flow-matching prior that refines the encoder's embedding.

Rectified flow from the encoder's prediction to the true embedding, both in
z-space, conditioned on the prediction:

    x0 = e_hat + sigma * eps,   x1 = z,   x_t = (1 - t) x0 + t x1
    L  = || v(x_t, t, e_hat) - (x1 - x0) ||^2

``e_hat`` is the encoder output put on the sphere of typical targets (the CLIP
loss only fixes its direction). At inference a few Euler steps go from e_hat.
Train it on encoder outputs of windows the encoder never saw (val): on its own
training windows the encoder is overfitted and the pairs are too easy.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from image_decoding.evaluation import retrieval
from image_decoding.nets import VectorUNet


@torch.no_grad()
def embed_split(model, loader):
    """Encoder outputs on a split for a new person: (e, y, subject)."""
    model.eval()
    device = model.mu.device
    e, y, s = [], [], []
    for X, target, subject in loader:
        e.append(model.encoder(model._select(X.to(device))).cpu()); y.append(target); s.append(subject)
    return torch.cat(e), torch.cat(y).numpy(), torch.cat(s).numpy()


def fit_prior(model, e, y, epochs=200, lr=3e-4, batch_size=512, sigma=0.1, monitor=None, n_steps=10, **unet_kwargs):
    """Attach a trained flow prior to ``model`` (an ImageDecoder) from pairs (encoder output e, target y).

    ``monitor = (e, y, subject)`` held out: the epoch with its best top-5 is kept.
    """
    device = model.mu.device
    model.prior, model.n_steps = VectorUNet(e.shape[1], **unet_kwargs).to(device), n_steps
    cond = model.on_sphere(e.to(device))
    z = (torch.as_tensor(y, dtype=torch.float32).to(device) - model.mu) / model.s
    opt = torch.optim.AdamW(model.prior.parameters(), lr=lr, weight_decay=1e-4)

    best, history = (-1.0, None), []
    for epoch in range(epochs):
        model.prior.train()
        for idx in torch.randperm(len(z), device=device).split(batch_size):
            c, x1 = cond[idx], z[idx]
            x0 = c + sigma * torch.randn_like(c)
            t = torch.rand(len(idx), device=device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            loss = F.mse_loss(model.prior(xt, t, c), x1 - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        row = {"epoch": epoch, "loss": loss.item()}
        if monitor is not None:
            row.update(retrieval(refined_prediction(model, monitor[0]), monitor[1], monitor[2]))
            if row["top5"] > best[0]:
                best = (row["top5"], copy.deepcopy(model.prior.state_dict()))
        history.append(row)
    if best[1] is not None:
        model.prior.load_state_dict(best[1])
    return history


@torch.no_grad()
def refined_prediction(model, e, n_steps=None):
    """Predicted targets y from encoder outputs e, with ``n_steps`` Euler steps (0 = no refinement)."""
    model.eval()
    keep = model.n_steps
    model.n_steps = keep if n_steps is None else n_steps
    z = e.to(model.mu.device)
    z = model.refine(z) if model.n_steps > 0 else z
    model.n_steps = keep
    return (model.mu + model.s * model.on_sphere(z)).cpu().numpy()


def split_by_image(y, fraction=0.1, seed=0):
    """Boolean mask holding out ``fraction`` of the images (all their windows)."""
    _, idx = np.unique(y, axis=0, return_inverse=True)
    idx = idx.reshape(-1)
    held = np.random.default_rng(seed).permutation(idx.max() + 1)[: max(1, int(fraction * (idx.max() + 1)))]
    return np.isin(idx, held)
