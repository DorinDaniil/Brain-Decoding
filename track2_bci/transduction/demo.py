"""Toy sessions, no data needed:  python demo.py

Every session has balanced classes whose centres are moved away from the ones
a "source" model knows. The source model's probabilities are the prior; the
transducer refines them window by window, seeing only the session so far.
"""

import argparse

import numpy as np

from method import Transducer


def session(rng, centres, n=150, shift=1.0, noise=2.0):
    """Balanced labels in random order; features around session-specific centres."""
    y = rng.permutation(np.arange(n) % len(centres))
    moved = centres + shift * rng.normal(size=centres.shape)
    return y, moved[y] + noise * rng.normal(size=(n, centres.shape[1]))


def source_proba(Z, centres, noise=2.0):
    d = ((Z[:, None] - centres[None]) ** 2).sum(-1) / (2 * noise ** 2)
    z = -d - (-d).max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def balanced_accuracy(y, pred):
    return np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--shift", type=float, default=1.0, help="how far session centres move from the source ones")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    centres = 0.6 * rng.normal(size=(3, args.dim))
    base, trans, late_base, late_trans = [], [], [], []
    for _ in range(args.sessions):
        y, Z = session(rng, centres, shift=args.shift)
        prior = source_proba(Z, centres)
        t = Transducer()
        q = np.array([t.step(z, p) for z, p in zip(Z, prior)])
        base.append(balanced_accuracy(y, prior.argmax(1)))
        trans.append(balanced_accuracy(y, q.argmax(1)))
        late_base.append(balanced_accuracy(y[100:], prior[100:].argmax(1)))
        late_trans.append(balanced_accuracy(y[100:], q[100:].argmax(1)))
    print(f"{args.sessions} sessions x 150 windows, 3 classes, chance 0.333")
    print(f"balanced accuracy, all windows:     source model {np.mean(base):.3f}   + transduction {np.mean(trans):.3f}")
    print(f"balanced accuracy, windows 100-150: source model {np.mean(late_base):.3f}   + transduction {np.mean(late_trans):.3f}")


if __name__ == "__main__":
    main()
