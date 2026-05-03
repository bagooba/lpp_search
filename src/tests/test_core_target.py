"""Tests for core/target.py"""
import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.target import Target, PipelineStage
from core.planet_candidate import PlanetCandidate, PType


@pytest.fixture
def sample_target(tmp_workdir):
    """A target in MERGED stage with a synthetic catalog."""
    root = tmp_workdir / "target_tic-999_gaiaID-GAIA999"
    root.mkdir(exist_ok=True)
    
    df = pd.DataFrame([{
        "TICID": 999, "Mass": 0.5, "Rad": 0.5,
        "aLSM": 0.2, "bLSM": 0.1, "Teff": 4000, "logg": 4.0
    }])
    df.to_csv(root / "tic_star_parameters.csv", index=False)
    
    t0 = Target(ticid=999, gaia_id="GAIA999", root_dir=root)
    t0.set_stage(PipelineStage.MERGED)
    return t0


def test_catalog_returns_dict():
    """target.catalog() returns the internal _catalog dict."""
    root = Path("/tmp/test_targets")
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id=None, root_dir=root)
    t0._catalog = {"aLSM": 0.2, "bLSM": 0.1, "Mass": 0.5}
    assert t0.catalog() == {"aLSM": 0.2, "bLSM": 0.1, "Mass": 0.5}


def test_save_load_state(tmp_workdir):  # fixture already has tmp_workdir
    """save_state() and load_state() round-trip correctly."""
    root = tmp_workdir / "target_tic-1"
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id="GAIA1", root_dir=root)
    t0.pipeline_stage = PipelineStage.MERGED
    t0.save_state()
    
    root2 = tmp_workdir / "target_tic-1"
    root2.mkdir(exist_ok=True)
    t1 = Target(ticid=1, gaia_id="GAIA1", root_dir=root2)
    t1.load_state()
    assert t1.pipeline_stage == PipelineStage.MERGED


def test_load_state_missing_file(tmp_workdir):
    """load_state() early returns when state file doesn't exist."""
    root = tmp_workdir / "target_tic-1"
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id="GAIA1", root_dir=root)
    t0.load_state()
    assert t0.pipeline_stage == PipelineStage.RAW


def test_ld_u1_u2_missing_raises(tmp_workdir):
    """Target.ld_u1_u2 raises ValueError when aLSM/bLSM not available."""
    root = tmp_workdir / "target_tic-1"
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id=None, root_dir=root)
    with pytest.raises(ValueError, match="Limb darkening not available"):
        t0.ld_u1_u2


def test_discover_ids_from_dirname(tmp_workdir):
    """Target.discover_ids_from_dirname() extracts TICID from path name."""
    d = tmp_workdir / "target_tic-42_gaiaID-GAIA42"
    d.mkdir(exist_ok=True)
    ticid, gaia_id = Target.discover_ids_from_dirname(d)
    assert ticid == 42
    assert gaia_id == "GAIA42"


def test_save_candidates(tmp_workdir):
    """save_candidates() writes candidates JSON."""
    root = tmp_workdir / "target_tic-1"
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id="GAIA1", root_dir=root)
    c0 = PlanetCandidate(ptype="Single", t0_days=10.0, period_days=5.0)
    t0.save_candidates("run_1", [c0])
    
    assert (t0.candidates_dir / "run_run_1.json").exists()
    state = json.loads((t0.state_path).read_text())
    assert state["last_run_id"] == "run_1"


def test_load_candidates_empty(tmp_workdir):
    """load_candidates() returns [] when file missing."""
    root = tmp_workdir / "target_tic-1"
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id="GAIA1", root_dir=root)
    assert t0.load_candidates() == []


def test_load_catalog_csv_missing(tmp_workdir):
    """load_catalog_csv() returns False when CSV not present."""
    root = tmp_workdir / "target_tic-1"
    root.mkdir(exist_ok=True)
    t0 = Target(ticid=1, gaia_id=None, root_dir=root)
    assert t0.load_catalog_csv() is False


class TestTargetCoverage:
    """Coverage tests for target.py missing paths."""
    
    def test_post_init_with_empty_catalog_row(self, tmp_workdir):
        """__post_init__ with empty catalog_row skips _catalog init."""
        root = tmp_workdir / "target_tic-1"
        root.mkdir(exist_ok=True)
        t0 = Target(ticid=1, gaia_id=None, root_dir=root)
        assert len(t0._catalog) == 0
    
    def test_post_init_populates_catalog(self, tmp_workdir):
        """__post_init initializes _catalog from catalog_row."""
        root = tmp_workdir / "target_tic-1"
        root.mkdir(exist_ok=True)
        row = pd.Series({"Mass": 0.5, "Rad": 0.5})
        t0 = Target(ticid=1, gaia_id=None, root_dir=root, catalog_row=row)
        assert "Mass" in t0._catalog
        assert t0.Mass == 0.5
    
    def test_rho_star_calculation(self, tmp_workdir):
        """_compute_rho_star_if_possible computes rho_star from Mass/Rad."""
        root = tmp_workdir / "target_tic-1"
        root.mkdir(exist_ok=True)
        t0 = Target(ticid=1, gaia_id=None, root_dir=root)
        t0.Mass = 0.5
        t0.Rad = 0.5
        t0._compute_rho_star_if_possible()
        assert np.isfinite(t0.rho_star)
    
    def test_stage_rank(self, tmp_workdir):
        """stage_rank() returns correct numeric rank."""
        root = tmp_workdir / "target_tic-1"
        root.mkdir(exist_ok=True)
        t0 = Target(ticid=1, gaia_id=None, root_dir=root)
        assert t0.stage_rank() == 0
        
        t0.set_stage(PipelineStage.MERGED)
        assert t0.stage_rank() == 2
    
    def test_stage_at_least(self, tmp_workdir):
        """stage_at_least() returns True if >= stage."""
        root = tmp_workdir / "target_tic-1"
        root.mkdir(exist_ok=True)
        t0 = Target(ticid=1, gaia_id=None, root_dir=root)
        assert t0.stage_at_least(PipelineStage.RAW) is True
        assert t0.stage_at_least(PipelineStage.MERGED) is False
        
        t0.set_stage(PipelineStage.MERGED)
        assert t0.stage_at_least(PipelineStage.MERGED) is True
        
        t0.set_stage(PipelineStage.SEARCHED)
        assert t0.stage_at_least(PipelineStage.MERGED) is True


def test_to_dict_from_dict():
    """PlanetCandidate to_dict/from_dict round-trip."""
    c0 = PlanetCandidate(ptype="Single", t0_days=10.0, period_days=5.0)
    d = c0.to_dict()
    c1 = PlanetCandidate.from_dict(d)
    assert c1.t0_days == c0.t0_days
    assert c1.period_days == c0.period_days
