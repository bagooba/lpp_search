# utils/queue.py
from pathlib import Path
import os

QUEUE_ROOT = Path(os.environ.get("PIPELINE_RUN_ROOT", "runs/current")) / "queue"

def enqueue(step: str, ticid: int) -> None:
    d = QUEUE_ROOT / step
    d.mkdir(parents=True, exist_ok=True)
    marker = d / str(int(ticid))
    try:
        marker.open("x").close()  # atomic create
    except FileExistsError:
        pass