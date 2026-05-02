# utils/target_lookup.py
from pathlib import Path
import glob
from core.target import Target

def find_target_dir_by_ticid(ticid: int, target_glob: str = "../toi_data/target_*") -> Path:
    ticid = int(ticid)
    for p in map(Path, sorted(glob.glob(target_glob))):
        t, _ = Target.discover_ids_from_dirname(p)  # already defined in Target class
        if int(t) == ticid:
            return p
    raise FileNotFoundError(f"No target dir found for TICID={ticid} under {target_glob}")


def load_ticids_txt(path: Path) -> list[int]:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return [int(x) for x in lines if x and not x.startswith("#")]

def resolve_ticid(argv) -> int:
    if "--ticid" in argv:
        i = argv.index("--ticid")
        return int(argv[i + 1])

    if "--ticid-file" in argv:
        i = argv.index("--ticid-file")
        ticids = load_ticids_txt(Path(argv[i + 1]))
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is None:
            raise SystemExit("Need SLURM_ARRAY_TASK_ID for --ticid-file, or pass --ticid.")
        return int(ticids[int(task)])

    # fallback: positional
    if len(argv) > 1:
        return int(argv[1])

    raise SystemExit("Usage: --ticid <TICID> OR --ticid-file <ticids.txt> (with SLURM_ARRAY_TASK_ID)")