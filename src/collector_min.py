
# collector_min.py
import json, pathlib, datetime

def collect(root="results"):
    root = pathlib.Path(root)
    rows, missing = [], []
    for td in sorted(p for p in root.iterdir() if p.is_dir()):
        sfile = td / "summary.json"
        if not sfile.exists():
            missing.append(td.name)
            continue
        with sfile.open() as f:
            s = json.load(f)
        rows.append({
            "target_id": s.get("target_id", td.name),
            "status": s.get("status"),
            "rhat_max": s.get("fit_quality", {}).get("rhat_max"),
            "ess_min": s.get("fit_quality", {}).get("ess_min"),
            "runtime_sec": s.get("runtime_sec")
        })
    payload = {
        "_schema": {"name": "run_index", "version": "1.0.0"},
        "run_metadata": {
            "created": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "num_targets": len(rows),
            "num_missing_summary": len(missing)
        },
        "targets": rows
    }
    with (root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if missing:
        with (root / "run_missing.json").open("w", encoding="utf-8") as f:
            json.dump({"missing": missing}, f, indent=2)

if __name__ == "__main__":
    collect()
