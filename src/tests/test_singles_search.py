# tests/test_singles_search.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from core.target import Target, PipelineStage
from stages.search_singles import singles_search, SinglesSearchConfig

def _make_synthetic_total_csv(root: Path, flavour: str = "TGLC",
                              t0=5.0, dur=0.25, depth=0.01, noise=8e-4) -> Path:
    """Write a small merged total CSV with one box-shaped dip."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 2000)  # ~7.2 min cadence
    f = np.ones_like(t) + rng.normal(0, noise, t.size)

    in_tr = np.abs(t - t0) < (dur / 2.0)
    f[in_tr] -= depth

    df = pd.DataFrame({
        "TIME": t.astype(float),
        "FLUX": f.astype(float),
        "FLUX_ERR": np.full_like(f, f.std(), dtype=float),
        "FLUX_TREND": np.zeros_like(f, dtype=float),
        "RAW_FLUX": f.astype(float),
        "RAW_FLUX_ERR": np.full_like(f, f.std(), dtype=float),
    })
    out = root / f"{root.name}_{flavour}_APER__total.csv"
    df.to_csv(out, index=False)
    return out

@pytest.fixture
def target_ready(tmp_workdir: Path) -> Target:
    """Temp target dir with MERGED stage and synthetic total CSV."""
    root = Path("target_tic-123456789_gaiaID-GAIA12345")
    root.mkdir(parents=True, exist_ok=True)
    t = Target(ticid=123456789, gaia_id="GAIA12345", root_dir=root)
    t.set_stage(PipelineStage.MERGED)
    # minimal catalog file (not used by singles, but harmless)
    pd.DataFrame([{"TICID": t.ticid, "Mass": 0.5, "Rad": 0.5, "Teff": 3800, "logg": 4.8}]
                 ).to_csv(root / "tic_star_parameters.csv", index=False)
    _make_synthetic_total_csv(root, flavour="TGLC")
    return t

def test_singles_search_finds_event_and_persists_state(monkeypatch, target_ready: Target):
    """
    Monkeypatch DT to emit one bbox near T0=5.0. Verify return frames and state persistence.
    """
    from stages import search_singles as ss

    def fake_DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat=True):
        # bbox layout: [score?, T0, center?, duration, height?]
        return [[0.9, 5.0, 1.0, 0.25, 0.99]]

    monkeypatch.setattr(ss, "DT_analysis", fake_DT_analysis)

    cfg = SinglesSearchConfig(flavour="TGLC", confidence=0.55, plot_events=False, verbose=False)
    planet_df, _ = singles_search(target_ready, cfg=cfg, run_1=True)

    # Returned dataframe sanity
    assert len(planet_df) == 1
    assert planet_df.loc[0, "TICID"] == target_ready.ticid
    assert pytest.approx(planet_df.loc[0, "T0"], rel=1e-3, abs=1e-3) == 5.0

    # State was persisted
    target_ready.load_state()
    assert target_ready.dt_prelim_found is True
    assert target_ready.quick_singles_t0 == [pytest.approx(5.0, rel=1e-3, abs=1e-3)]

def test_singles_search_unique_sorted_t0(monkeypatch, target_ready: Target):
    """
    If DT returns duplicate/out-of-order events, Target.quick_singles_t0 should be unique+sorted.
    """
    from stages import search_singles as ss

    def fake_DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat=True):
        return [
            [0.9, 9.0, 1.0, 0.2, 0.99],
            [0.9, 3.0, 1.0, 0.2, 0.99],
            [0.9, 3.0, 1.0, 0.2, 0.99],  # duplicate
            [0.9, 6.0, 1.0, 0.2, 0.99],
        ]

    monkeypatch.setattr(ss, "DT_analysis", fake_DT_analysis)

    cfg = SinglesSearchConfig(flavour="TGLC", confidence=0.55)
    planet_df, _ = singles_search(target_ready, cfg=cfg, run_1=True)

    assert list(planet_df["T0"]) == [3.0, 6.0, 9.0]
    target_ready.load_state()
    assert target_ready.quick_singles_t0 == [3.0, 6.0, 9.0]

def test_singles_search_no_total_csv_errors(tmp_workdir: Path):
    """
    Without any merged total CSV present, singles_search should raise FileNotFoundError.
    """
    root = Path("target_tic-1_gaiaID-NA")
    root.mkdir(parents=True, exist_ok=True)
    t = Target(ticid=1, gaia_id=None, root_dir=root)

    with pytest.raises(FileNotFoundError):
        singles_search(t)

def test_singles_search_no_signal_sets_false(monkeypatch, target_ready: Target):
    """
    If DT yields no boxes, dt_prelim_found=False and quick_singles_t0=[] are persisted.
    """
    from stages import search_singles as ss
    monkeypatch.setattr(ss, "DT_analysis", lambda *_, **__: [])

    cfg = SinglesSearchConfig(flavour="TGLC", confidence=0.55)
    planet_df, _ = singles_search(target_ready, cfg=cfg, run_1=True)

    assert len(planet_df) == 0
    target_ready.load_state()
    assert target_ready.dt_prelim_found is False
    assert target_ready.quick_singles_t0 == []