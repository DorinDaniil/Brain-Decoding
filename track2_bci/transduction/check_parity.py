"""Check that method.py computes what bci/stream.py ships in the submission:  python check_parity.py

Same synthetic session through both: tangent features window by window, then the
transducer's output for every window. Needs the ``bci`` package next to this folder.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bci import stream as shipped  # noqa: E402
from bci.data import SFREQ, synthetic  # noqa: E402

from method import BANDS, TangentFeatures, Transducer  # noqa: E402


def main():
    windows = synthetic(n_subjects=2, multi_session=0, runs=4, per_run=15)
    train, test = windows.X[windows.subject == "sub-0"], windows.X[windows.subject == "sub-1"]
    log_priors = TangentFeatures.fit_log_priors(train, SFREQ)

    feats, trans = TangentFeatures(SFREQ, log_priors=log_priors), Transducer()
    sos = [shipped.bandpass(b, SFREQ) for b in BANDS]
    states = [[5 * lp, 5] for lp in log_priors]
    rng = np.random.default_rng(0)
    probas = rng.dirichlet(np.ones(3), size=len(test))

    z_gap, q_gap, Z = 0.0, 0.0, []
    for i, (x, p) in enumerate(zip(test, probas)):
        z_ref = np.concatenate([shipped.tangent_step(s, shipped.covariances(x[None], f)[0]) for s, f in zip(states, sos)])
        z = feats.step(x)
        q = trans.step(z, p)
        Z.append(z_ref)
        if i + 1 >= trans.warm:
            Zs = np.array(Z)
            q_ref = shipped.self_label(Zs @ Zs.T, np.log(probas[: i + 1] + 1e-9))[-1]
        else:
            q_ref = p
        z_gap, q_gap = max(z_gap, np.abs(z - z_ref).max()), max(q_gap, np.abs(q - q_ref).max())
    print(f"{len(test)} windows: max |features - shipped| {z_gap:.1e}, max |probabilities - shipped| {q_gap:.1e}")
    assert z_gap < 1e-8 and q_gap < 1e-6, "method.py and bci/stream.py disagree"
    print("OK")


if __name__ == "__main__":
    main()
