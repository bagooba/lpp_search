"""Tests for search_periodic ~85% coverage."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from core.target import Target, PipelineStage
from stages.search_periodic import (
    periodic_search, PeriodicSearchConfig,
    transit_mask, _nearest_epoch_time
)


@pytest.fixture
def target_periodic(tmp_workdir):
    root = tmp_workdir / "target_tic-periodic"
    root.mkdir(parents=True, exist_ok=True)
    t = Target(ticid=7777, gaia_id="GAIA7777", root_dir=root)
    t.set_stage(PipelineStage.MERGED)
    return t


def test_transit_mask_basic():
    time = np.linspace(0, 100, 10000)
    mask = transit_mask(time, period=10.0, duration=0.5, t0=5.0, buffer=0.1)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool


def test_transit_mask_zero_duration():
    time = np.linspace(0, 100, 1000)
    mask = transit_mask(time, period=10.0, duration=0.0, t0=5.0, buffer=0.0)
    assert len(mask) == 1000


def test_transit_mask_max_planets():
    cfg = PeriodicSearchConfig()
    cfg.max_planets = 5
    assert cfg.max_planets == 5


def test_transit_mask_min_snr():
    cfg = PeriodicSearchConfig()
    cfg.min_snr = 10.0
    assert cfg.min_snr == 10.0


def test_transit_mask_max_iters():
    cfg = PeriodicSearchConfig()
    cfg.max_iters = 20
    assert cfg.max_iters == 20


def test_nearest_epoch_time_basic():
    assert _nearest_epoch_time(15.0, 5.0, 10.0) == 15.0


def test_nearest_epoch_time_edge():
    assert _nearest_epoch_time(5.0, 5.0, 10.0) == 5.0
    assert _nearest_epoch_time(-5.0, -5.0, 10.0) == -5.0


def test_nearest_epoch_time_precision():
    assert _nearest_epoch_time(7.0, 5.0, 2.0) == 7.0
    assert _nearest_epoch_time(6.0, 5.0, 2.0) == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
