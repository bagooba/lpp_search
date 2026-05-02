# utils/queue.py
from pathlib import Path
import os

# QUEUE_ROOT = Path(os.environ.get("PIPELINE_QUEUE_ROOT", "runs/queue"))

QUEUE_ROOT = Path(
    os.environ.get("PIPELINE_RUN_ROOT", "runs/current")
) / "queue"

def enqueue(step: str, ticid: int) -> None:
    d = os.path.join(QUEUE_ROOT, step)
    d.mkdir(parents=True, exist_ok=True)
    marker =  os.path.join(d, str(int(ticid)))
    try:
        marker.open("x").close()
    except FileExistsError:
        pass
