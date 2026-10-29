"""Stage 2 (optional): a flow-matching prior that refines the ensemble's prediction.

Rectified flow, in z = (y - mu) / s, from the prediction to the true embedding,
conditioned on the prediction:

    c = (radius * p / |p| - mu) / s,   x0 = c + sigma * eps,   x1 = z,
    x_t = (1 - t) x0 + t x1,           L = || v(x_t, t, c) - (x1 - x0) ||^2

It is trained on windows the encoders never saw: half of the held-out subjects,
on the training images. The number of Euler steps, 0 included (no prior), is
chosen by the subject-aggregated top-5 of the other half on the test images
(new people for the prior too), so the prior only ships when it helps.
"""

import copy

import torch
import torch.nn.functional as F

from image_decoding.evaluation import retrieval_ids
from image_decoding.nets import VectorUNet


def fit_prior(model, p, y, epochs=100, lr=3e-4, batch_size=512, sigma=0.1, monitor=None, n_steps=10, **unet_kwargs):
    """Attach a trained flow prior to ``model`` (an ImageDecoder) from pairs (prediction p, target y).

    ``monitor = (p, image, subject, images)`` held out: the epoch with its best
    subject-aggregated top-5 is kept.
    """
    device = model.mu.device
    model.prior, model.n_steps = VectorUNet(p.shape[1], **unet_kwargs).to(device), n_steps
    cond = (model.radius * F.normalize(torch.as_tensor(p, device=device), dim=-1) - model.mu) / model.s
    z = (torch.as_tensor(y, dtype=torch.float32, device=device) - model.mu) / model.s
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
            row.update(score_steps(model, *monitor))
            if row["top5_subject_agg"] > best[0]:
                best = (row["top5_subject_agg"], copy.deepcopy(model.prior.state_dict()))
        history.append(row)
    if best[1] is not None:
        model.prior.load_state_dict(best[1])
    return history


@torch.no_grad()
def refined(model, p, n_steps, batch_size=8192):
    """Predictions p (N, D) after ``n_steps`` Euler steps of the prior (0: unchanged)."""
    if n_steps == 0 or model.prior is None:
        return p
    model.eval()
    keep, model.n_steps = model.n_steps, n_steps
    out = torch.cat([model.refine(torch.as_tensor(p[i:i + batch_size], device=model.mu.device)).cpu()
                     for i in range(0, len(p), batch_size)])
    model.n_steps = keep
    return out.numpy()


def score_steps(model, p, image, subject, images, n_steps=None):
    return retrieval_ids(refined(model, p, model.n_steps if n_steps is None else n_steps), image, subject, images)
