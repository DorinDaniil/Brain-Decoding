"""Riemann + manifold gate.

Every training subject k has a center c_k on the manifold (mean session
fingerprint) and a learnable embedding e_k. A session's embedding mixes the
embeddings of the nearest subjects:

    alpha_k = softmax_k(-||fp - c_k||^2 / tau),   e = sum_k alpha_k e_k   (tau is learned)

and e modulates the classifier through FiLM. During training a window's own
center is hidden from the gate with probability ``hide_own``, so the model
also learns to work through neighbours, as it must for a new person.
"""

import numpy as np
import torch
import torch.nn as nn

from bci.evaluation import score
from bci.riemann import RiemannFeatures, StreamingDecoder

T = lambda a: torch.as_tensor(a, dtype=torch.float32)


class ManifoldGatedRiemann(nn.Module):
    def __init__(self, n_feat, centers, tau, n_classes=3, dim=32, emb=8):
        super().__init__()
        self.register_buffer("centers", T(centers))        # where training subjects lie
        self.table = nn.Embedding(len(centers), emb)        # learnable embedding of each
        self.log_tau = nn.Parameter(T(np.log(tau)))         # gate temperature, learned
        self.proj, self.drop = nn.Linear(n_feat, dim), nn.Dropout(0.3)
        self.film = nn.Linear(emb, 2 * dim)
        self.head = nn.Linear(dim, n_classes)

    def gate(self, fp, hide=None):
        """Weights of the training subjects for a session with fingerprint fp."""
        logits = -torch.cdist(fp, self.centers) ** 2 / self.log_tau.exp()
        if hide is not None:
            logits = logits.masked_fill(hide, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(self, x, fp, hide=None):
        e = self.gate(fp, hide) @ self.table.weight
        gamma, beta = self.film(e).chunk(2, dim=-1)
        return self.head(torch.relu(self.proj(self.drop(x)) * (1 + gamma) + beta))


def fit(train, epochs=150, lr=1e-3, hide_own=0.5, dim=32, emb=8,
        band=(4, 40), prior_weight=5, seed=0, val=None):
    """Train on ``train``; with ``val``, ``model.history`` holds val session_mean per epoch."""
    torch.manual_seed(seed)
    features = RiemannFeatures(band, prior_weight)
    x, fp = features.fit_transform(train)
    subjects = sorted(set(train.subject))
    s = np.array([subjects.index(v) for v in train.subject])
    centers = np.stack([fp[s == k].mean(0) for k in range(len(subjects))])
    d2 = ((centers[:, None] - centers[None]) ** 2).sum(-1)

    net = ManifoldGatedRiemann(x.shape[1], centers, tau=np.median(d2[d2 > 0]), dim=dim, emb=emb)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    weights = T(len(s) / (3 * np.bincount(train.y, minlength=3)))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    X, F, Y = T(x), T(fp), torch.as_tensor(train.y)
    own = torch.nn.functional.one_hot(torch.as_tensor(s), len(subjects)).bool()
    if val is not None:
        x_va, fp_va = features.transform(val)

    history = []
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        hide = own & (torch.rand(len(s), 1) < hide_own)  # sometimes hide the subject itself
        loss_fn(net(X, F, hide), Y).backward(); opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad():
                history.append(score(val, net(T(x_va), T(fp_va)).argmax(-1).numpy())["session_mean"])

    net.eval()

    def classify(x, fp):
        with torch.no_grad():
            return int(net(T(x[None]), T(fp[None])).argmax())

    model = StreamingDecoder(features, classify)
    model.net, model.subjects, model.history = net, subjects, history
    return model
