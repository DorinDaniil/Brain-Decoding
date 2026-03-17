from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_ID = "ds004840"
DATASET_GIT = "https://github.com/OpenNeuroDatasets/ds004840.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download OpenNeuro dataset ds004840 (BIDS) using datalad.")
    parser.add_argument("--out_dir", type=Path, default=Path("data") / DATASET_ID)
    parser.add_argument("--version", type=str, default="1.0.1", help="Git tag/branch to checkout (default: 1.0.1).")
    args = parser.parse_args(argv)

    if shutil.which("datalad") is None:
        raise RuntimeError(
            "datalad is not installed.\n\n"
            "Install (Python): python -m pip install datalad\n"
            "Also install Git and git-annex:\n"
            "  - Git: https://git-scm.com/download/win\n"
            "  - git-annex: https://git-annex.branchable.com/install/windows/\n"
        )

    out_dir = args.out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if not out_dir.exists():
        run(["datalad", "install", "-s", DATASET_GIT, str(out_dir)])

    # Optional: pin the OpenNeuro version (git tag). If it fails, user can delete this line.
    run(["git", "checkout", args.version], cwd=out_dir)

    # Download all annexed files.
    run(["datalad", "get", "-r", "."], cwd=out_dir)

    print(f"\nOK. Dataset is in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

