# tests/test_run_quick_singles.py
from pathlib import Path
import importlib.util
import json
import os
import glob
import pandas as pd
import pytest

from core.target import Target, PipelineStage

def _load_runner_module() -> object:
    """
    Load 'scripts/02_run_quick_singles.py' as a module regardless of digit-leading filename.
    Returns the loaded module object which exposes 'main', 'TARGET_GLOB', etc.
    """
    script_path = Path("scripts") / "02_run_quick_singles.py"
    assert script_path.exists(), f"Missing script: {script_path}"
    spec = importlib.util.spec_from_file_location("runner02", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def _make_synthetic_total_csv(root: Path, flavour: str = "TGLC",
                              t0=5.0, dur=0.25, depth=0.01) -> Path:
    import numpy as np
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 2000)
    f = np.ones_like(t) + rng.normal(0, 8e-4, t.size)
    in_tr = np.abs(t - t0) < (dur / 2.0)
    f[in_tr] -= depth
    df = pd.DataFrame({
        "TIME": t, "FLUX": f, "FLUX_ERR": f.std(),
        "FLUX_TREND": 0.0, "RAW_FLUX": f, "RAW_FLUX_ERR": f.std()
    })
    out = root / f"{root.name}_{flavour}_APER__total.csv"
    df.to_csv(out, index=False)
    return out

@pytest.fixture
def prepared_targets(tmp_workdir: Path):
    """
    Build two targets:
      - target 0: MERGED + merged total CSV (should run)
      - target 1: RAW, no total CSV (should be skipped)
    """
    # target 0
    t0_root = Path("target_tic-100_gaiaID-GAIA100"); t0_root.mkdir()
    t0 = Target(ticid=100, gaia_id="GAIA100", root_dir=t0_root)
    t0.set_stage(PipelineStage.MERGED)
    # minimal catalog (optional)
    pd.DataFrame([{"TICID": t0.ticid, "Mass": 0.5, "Rad": 0.5, "Teff": 3800, "logg": 4.8}]
                 ).to_csv(t0_root / "tic_star_parameters.csv", index=False)
    _make_synthetic_total_csv(t0_root, flavour="TGLC")

    # target 1 (not ready)
    t1_root = Path("target_tic-101_gaiaID-GAIA101"); t1_root.mkdir()
    t1 = Target(ticid=101, gaia_id="GAIA101", root_dir=t1_root)
    t1.save_state()

    return [t0_root, t1_root]

def test_runner_main_detects_and_updates_state(monkeypatch, prepared_targets):
    runner = _load_runner_module()

    # Restrict the glob to the temp fixtures
    monkeypatch.setattr(runner, "TARGET_GLOB", "target_*", raising=True)

    # Stub DT to guarantee one detection at T0=5.0
    from stages import search_singles as ss
    monkeypatch.setattr(ss, "DT_analysis",
        lambda *_, **__: [[0.9, 5.0, 1.0, 0.25, 0.99]], raising=True
    )

    # Run index 0 (MERGED)
    runner.main(0)

    state = json.loads((prepared_targets[0] / "target.state.json").read_text())
    assert state["pipeline_stage"] == "SEARCHED"
    assert state["dt_prelim_found"] is True
    assert isinstance(state["quick_singles_t0"], list) and len(state["quick_singles_t0"]) == 1

def test_runner_skips_when_not_ready(monkeypatch, prepared_targets, capsys):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "TARGET_GLOB", "target_*", raising=True)

    # Run index 1 (RAW + no total file)
    runner.main(1)
    out = capsys.readouterr().out
    assert "Not ready (PipelineStage < MERGED and no *total.csv). Skipping." in out

def test_runner_respects_slurm_array_env(monkeypatch, prepared_targets):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "TARGET_GLOB", "target_*", raising=True)

    # Stub DT again so detection occurs
    from stages import search_singles as ss
    monkeypatch.setattr(ss, "DT_analysis",
        lambda *_, **__: [[0.9, 4.0, 1.0, 0.20, 0.99]], raising=True
    )

    # Emulate SLURM setting; we still call main(int(env)) explicitly in test
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    runner.main(int(os.environ["SLURM_ARRAY_TASK_ID"]))

    state = json.loads((prepared_targets[0] / "target.state.json").read_text())
    assert state["dt_prelim_found"] is True
    assert len(state["quick_singles_t0"]) == 1