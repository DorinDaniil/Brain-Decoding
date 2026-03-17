### Scripts

- `download_ds004840.py`: download the OpenNeuro dataset `ds004840` (BIDS).

Usage:

```bash
python scripts/download_ds004840.py --out_dir data/ds004840 --version 1.0.1
```

Notes:

- **Recommended**: `datalad` (keeps dataset as a git-annex repo; good for large files and reproducibility).
- Windows prerequisites:
  - Git: `https://git-scm.com/download/win`
  - git-annex: `https://git-annex.branchable.com/install/windows/`
  - datalad: `python -m pip install datalad`

