"""Training and evaluation of PoseNet.

Metric as on the leaderboard: absolute angle error over every joint and time
point, in degrees; ranking by its mean over subjects (each subject's MAE first).

Augmentations, all label-preserving for cross-user transfer:
- band rotation: the electrode ring shifted by -1, 0 or +1 per window;
- per-electrode gain, log-uniform in [0.5, 2] (skin contact, anatomy);
- additive noise and two time masks per window.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from emg_pose.nets import PoseNet

DEG = 180 / math.pi


def augment(X, max_shift=1, gain=2.0, noise=0.05, n_masks=2, mask_len=400):
    B, C, T = X.shape
    shift = torch.randint(-max_shift, max_shift + 1, (B,), device=X.device)
    index = (torch.arange(C, device=X.device)[None] - shift[:, None]) % C
    X = X.gather(1, index[:, :, None].expand(-1, -1, T))
    X = X * torch.exp(torch.empty(B, C, 1, device=X.device).uniform_(-math.log(gain), math.log(gain)))
    X = X + noise * X.std(dim=-1, keepdim=True) * torch.randn_like(X)
    for _ in range(n_masks):
        start = torch.randint(0, T - mask_len, (B, 1), device=X.device)
        t = torch.arange(T, device=X.device)[None]
        X = X * ~((t >= start) & (t < start + mask_len))[:, None]
    return X


def mean_pose(loader, n_batches=200):
    total, count = 0.0, 0
    for (_, y, _), _ in zip(loader, range(n_batches)):
        total = total + y.sum(dim=(0, 2))
        count += y.shape[0] * y.shape[2]
    return total / count


@torch.no_grad()
def evaluate(model, loader):
    """{"mae", "subject_mean"} in degrees for a model with ``predict(X) -> (B, J, T)`` radians."""
    sums, counts = {}, {}
    for X, y, subject in loader:
        pred = torch.as_tensor(model.predict(X)).float().cpu()
        if pred.shape[-1] != y.shape[-1]:  # the scorer's nearest-neighbor resampling
            pred = F.interpolate(pred, size=y.shape[-1], mode="nearest")
        err = (pred - y).abs().mean(dim=(1, 2))
        for e, s in zip(err.tolist(), torch.as_tensor(subject).tolist()):
            sums[s] = sums.get(s, 0.0) + e
            counts[s] = counts.get(s, 0) + 1
    per_subject = np.array([sums[s] / counts[s] for s in sums])
    windows = sum(counts.values())
    return {"mae": float(DEG * sum(sums.values()) / windows), "subject_mean": float(DEG * per_subject.mean())}


class _Single:
    """predict() for one PoseNet during training."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        return self.net.angles(X.to(self.net.mean_pose.device))


def train(loaders, meta, epochs=40, lr=1e-3, weight_decay=1e-2, device="cpu", seed=0, log=print, **net_kwargs):
    """Train one PoseNet with L1 on the angles; keep the epoch with the best val subject_mean."""
    torch.manual_seed(seed)
    net = PoseNet(meta["n_chans"], meta["n_joints"], meta["sfreq"], mean_pose=mean_pose(loaders["train"]),
                  **net_kwargs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    use_amp = device.startswith("cuda")

    best, history = (math.inf, None), []
    for epoch in range(epochs):
        net.train()
        for X, y, _ in loaders["train"]:
            X, y = augment(X.to(device, non_blocking=True)), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = net(X)
            pred = F.interpolate(pred.float(), size=y.shape[-1], mode="linear", align_corners=True)
            loss = (pred - y).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        val = evaluate(_Single(net), loaders["val"])
        history.append({"epoch": epoch, "loss": DEG * loss.item(), **val})
        log(history[-1])
        if val["subject_mean"] < best[0]:
            best = (val["subject_mean"], copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])
    return net, history
