"""Train the components of the streaming model (``bci.stream``) and assemble variants.

Components, each optional in a variant:

- ``forest``: Riemannian random forest over everyone (bands, channel subsets,
  bootstrap of subjects), on the base bands or on the extended bank that adds
  0.5-4 Hz (slow, ocular), 40-58 Hz (muscle, high gamma) and 1-58 Hz;
- ``subjects``: a small forest per training participant, recentered from that
  participant's own mean;
- ``router``: session means of the training participants and a ratio-test
  threshold calibrated on them (``subjects`` need it);
- ``deep``: ShallowConvNet seeds on causally aligned windows;
- ``balance``: no training, two numbers.

    parts = fit_components(train, list(VARIANTS.values()))   # shared parts, trained once
    spec, model = assemble(parts, VARIANTS["all"], train.ch_names)
"""

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import sosfiltfilt
from sklearn.linear_model import LogisticRegression

from bci.data import SFREQ
from bci.stream import ShallowNet, StreamModel, align_step, bandpass, covariances, logm, tangent_step

BASE_BANDS = ((4, 40), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40))
EXTRA_BANDS = ((0.5, 4), (40, 58), (1, 58))
ROUTER_BAND = (1, 40)
BANDS = BASE_BANDS + EXTRA_BANDS + (ROUTER_BAND,)  # the spec's band bank; experts refer to it by index
DEEP_BAND = (1, 58)

# variant name -> switches (missing = off / default)
VARIANTS = {
    "forest": {},
    "+balance": {"balance": True},
    "+bands": {"bands": "extended"},
    "+routing": {"routing": True},
    "+deep": {"deep": True},
    "all but deep": {"bands": "extended", "routing": True, "balance": True},
    "all": {"bands": "extended", "routing": True, "deep": True, "balance": True},
}


class Covariances:
    """Covariances of every window in every band of the bank, computed on first use."""

    def __init__(self, windows):
        self.windows, self.cache = windows, {}

    def __getitem__(self, band):
        if band not in self.cache:
            self.cache[band] = covariances(self.windows.X, bandpass(BANDS[band], SFREQ))
        return self.cache[band]


def fit_expert(windows, covs, band, channels, group, C=0.01, prior_weight=5, weight=None):
    """One Riemannian expert trained on all of ``windows`` (rows weighted by ``weight``)."""
    sub = covs[band][:, channels][:, :, channels]
    logs = np.stack([logm(c) for c in sub])
    log_prior = logs.mean(0)
    x = np.empty((len(windows), len(channels) * (len(channels) + 1) // 2))
    for idx in windows.streams():
        state = [prior_weight * log_prior, prior_weight]
        for i in idx:
            x[i] = tangent_step(state, sub[i], logs[i])
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1.0
    weight = np.ones(len(windows)) if weight is None else weight
    keep = weight > 0
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(((x - mean) / scale)[keep], windows.y[keep], sample_weight=weight[keep])
    return {"band": int(band), "channels": np.asarray(channels), "group": int(group),
            "log_prior": log_prior, "prior_weight": prior_weight, "x_mean": mean, "x_scale": scale,
            "coef": clf.coef_, "intercept": clf.intercept_}


def fit_forest(windows, covs, bands, n_trees=50, n_channels=20, C=0.01, prior_weight=5, group=-1,
               bootstrap_subjects=True, seed=0):
    rng = np.random.default_rng(seed)
    subjects = sorted(set(windows.subject))
    experts = []
    for _ in range(n_trees):
        band = int(rng.choice(bands))
        channels = np.sort(rng.choice(windows.X.shape[1], n_channels, replace=False))
        weight = None
        if bootstrap_subjects and len(subjects) > 1:
            drawn = Counter(rng.choice(subjects, len(subjects)))
            weight = np.array([drawn[s] for s in windows.subject], float)
        experts.append(fit_expert(windows, covs, band, channels, group, C, prior_weight, weight))
    return experts


def fit_router(windows, covs, subjects, min_windows=10, sharpness=0.05):
    """Session means of the training participants and the ratio-test threshold.

    ratio = (distance to the nearest participant) / (to the second nearest), of a
    session mean against the other sessions' means. Impostors: every session
    against the other participants only (a new person). Genuine: sessions of
    participants with several sessions against the rest (a later session of a
    known person). The threshold sits between the two.
    """
    band = BANDS.index(ROUTER_BAND)
    logs = np.stack([logm(c) for c in covs[band]])
    stream = windows.stream
    keys = list(dict.fromkeys(stream))
    templates = np.stack([logs[stream == k].mean(0) for k in keys])
    owner = np.array([subjects.index(windows.subject[stream == k][0]) for k in keys])

    def ratio(i, allowed):
        d = np.linalg.norm(templates - templates[i], axis=(1, 2))
        per_subject = sorted(d[allowed & (owner == k)].min() for k in set(owner[allowed]))
        own = d[allowed & (owner == owner[i])]
        if own.size:  # genuine: the own subject against the nearest other one
            other = min(d[allowed & (owner == k)].min() for k in set(owner[allowed]) - {owner[i]})
            return own.min() / other
        return per_subject[0] / per_subject[1]

    every = np.ones(len(keys), bool)
    impostor = np.array([ratio(i, owner != owner[i]) for i in range(len(keys))])
    genuine = np.array([ratio(i, every & (np.arange(len(keys)) != i)) for i in range(len(keys))
                        if (owner == owner[i]).sum() > 1])
    if genuine.size and genuine.max() < impostor.min():
        threshold, width = (genuine.max() + impostor.min()) / 2, (impostor.min() - genuine.max()) / 8
    elif genuine.size:
        threshold, width = (np.median(genuine) + np.median(impostor)) / 2, 0.02
    else:
        threshold, width = impostor.min() - 0.05, 0.02
    print(f"router: impostor ratios {np.round(np.sort(impostor), 2)}, genuine {np.round(np.sort(genuine), 2)}"
          f" -> threshold {threshold:.3f}")
    return {"band": band, "templates": np.stack([np.asarray(t) for t in templates]), "template_subject": owner,
            "n_subjects": len(subjects), "ratio": float(threshold), "width": float(max(width, 1e-3)),
            "min_windows": min_windows, "sharpness": sharpness}


def aligned_windows(windows, Xf, cov_prior, prior_weight):
    """Band-passed windows ``Xf`` aligned causally within each session, as at inference."""
    out = np.empty_like(Xf)
    for idx in windows.streams():
        state = [prior_weight * cov_prior, prior_weight]
        for i in idx:
            out[i] = align_step(state, Xf[i])
    return out


def fit_deep(windows, seeds=(0,), epochs=100, lr=1e-3, weight_decay=1e-2, batch_size=64, prior_weight=5,
             device=None, log=print):
    """ShallowConvNet seeds on aligned windows -> the spec's ``deep`` entry (its mixing weight set later)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xf = sosfiltfilt(bandpass(DEEP_BAND, SFREQ), np.asarray(windows.X, np.float64), axis=-1)
    cov_prior = np.mean([x @ x.T / x.shape[-1] for x in Xf], axis=0)
    X = torch.as_tensor(aligned_windows(windows, Xf, cov_prior, prior_weight), dtype=torch.float32)
    y = torch.as_tensor(windows.y)
    class_weight = torch.as_tensor(len(y) / (3 * np.bincount(windows.y, minlength=3)), dtype=torch.float32)
    nets = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = ShallowNet(X.shape[1], X.shape[2]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for epoch in range(epochs):
            net.train()
            for idx in torch.randperm(len(y)).split(batch_size):
                loss = F.cross_entropy(net(X[idx].to(device)), y[idx].to(device), weight=class_weight.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            schedule.step()
        log(f"deep seed {seed}: final training loss {loss.item():.3f}")
        nets.append({"config": net.config, "state": {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}})
    return {"band": DEEP_BAND, "prior_weight": prior_weight, "cov_prior": cov_prior, "nets": nets}


def fit_components(train, switches, n_trees=50, n_subject_trees=10, n_channels=20, C=0.01, prior_weight=5,
                   deep_seeds=(0,), deep_epochs=100, seed=0, log=print):
    """Everything the variants in ``switches`` (a list of switch dicts) need, trained once on ``train``."""
    covs = Covariances(train)
    banks = {"base": list(range(len(BASE_BANDS))), "extended": list(range(len(BASE_BANDS) + len(EXTRA_BANDS)))}
    needed = {s.get("bands", "base") for s in switches}
    parts = {}
    for name in needed:
        parts[f"forest/{name}"] = fit_forest(train, covs, banks[name], n_trees, n_channels, C, prior_weight, seed=seed)
    if any(s.get("routing") for s in switches):
        subjects = sorted(set(train.subject))
        for name in {s.get("bands", "base") for s in switches if s.get("routing")}:
            experts = []
            for k, s in enumerate(subjects):
                mine = train.select(train.subject == s)
                experts += fit_forest(mine, Covariances(mine), banks[name], n_subject_trees, n_channels, C,
                                      prior_weight, group=k, bootstrap_subjects=False, seed=seed + 1 + k)
            parts[f"subjects/{name}"] = experts
        parts["router"] = fit_router(train, covs, subjects)
    if any(s.get("deep") for s in switches):
        parts["deep"] = fit_deep(train, deep_seeds, deep_epochs, prior_weight=prior_weight, log=log)
    return parts


def assemble(parts, switches, ch_names, deep_weight=0.3, alpha=1.0, prior_windows=10, deep_pool="log"):
    """A StreamModel spec for ``switches`` from trained ``parts``, and the model itself."""
    bands = switches.get("bands", "base")
    spec = {"sfreq": SFREQ, "ch_names": [str(c) for c in ch_names], "bands": [list(b) for b in BANDS],
            "experts": list(parts[f"forest/{bands}"]), "router": None, "deep": None, "balance": None}
    if switches.get("routing"):
        spec["experts"] += parts[f"subjects/{bands}"]
        spec["router"] = parts["router"]
    if switches.get("deep"):
        spec["deep"] = dict(parts["deep"], weight=deep_weight, pool=deep_pool)
    if switches.get("balance"):
        spec["balance"] = {"alpha": alpha, "prior_windows": prior_windows}
    return spec, StreamModel(spec)
