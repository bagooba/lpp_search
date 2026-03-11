
# scripts/01_prepare_targets.py
from pathlib import Path
import glob
import time as tm
import pandas as pd

from core.target import Target
from stages.dataprep import DataPrep

def prepare_one_target(target_dir: Path) -> None:
    ticid, gaia_id = Target.discover_ids_from_dirname(target_dir)
    dummy = pd.Series(dtype=object)
    t = Target(ticid=ticid, gaia_id=gaia_id, root_dir=target_dir, catalog_row=dummy)
    t.load_state()  # safe if re-running

    dp = DataPrep(target=t, flavour="TGLC")
    total_file = dp.prepare()
    print(f"Prepared TIC {t.ticid}: {total_file}")

if __name__ == "__main__":
    t0 = tm.time()
    target_dirs = sorted(glob.glob("../oi_data/target_*"))  # adjust your path
    print("num files", len(target_dirs))
    for td in target_dirs:
        try:
            prepare_one_target(Path(td))
        except Exception as e:
            print(f"[WARN] {td}: {e}")
    t1 = tm.time()
    print("time it took:", (t1 - t0)/60, "minutes")
