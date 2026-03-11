# helpers_io_min.py
import json, time
from pathlib import Path
from datetime import datetime

def iso_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

class Timer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time()
    def stop(self): return 0.0 if self.t0 is None else max(0.0, time.time() - self.t0)