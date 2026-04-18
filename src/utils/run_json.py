import json

def upsert_run_json(run_path, payload_update):
    if run_path.exists():
        base = json.loads(run_path.read_text())
    else:
        base = {}
    base.update(payload_update)
    run_path.write_text(json.dumps(base, indent=2))