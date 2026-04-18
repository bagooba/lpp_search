# utils/find_total_csv.py
from pathlib import Path

def find_total_csv(root_dir: Path, flavour: str) -> Path:
    patt = f"*{flavour}*_*total.csv"
    m = sorted(root_dir.glob(patt))
    if m:
        return max(m, key=lambda p: p.stat().st_mtime)

    m = sorted(root_dir.glob("*total.csv"))
    if m:
        return max(m, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"No merged total CSV found in {root_dir}.")
