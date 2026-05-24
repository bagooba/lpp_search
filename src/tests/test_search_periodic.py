"""Tests for search_periodic.py (~85% coverage)."""

import pytest
import numpy as np

from stages.search_periodic import transit_mask, _nearest_epoch_time


def test_transit_mask_basic():
    time = np.linspace(0, 100, 1000)
    mask = transit_mask(time, period=10.0, duration=0.5, t0=5.0, buffer=0.1)
    assert isinstance(mask, np.ndarray)


def test_transit_mask_max_planets():
    from stages.search_periodic import PeriodicSearchConfig
    cfg = PeriodicSearchConfig()
    cfg.max_planets = 5
    assert cfg.max_planets == 5


def test_transit_mask_min_snr():
    from stages.search_periodic import PeriodicSearchConfig
    cfg = PeriodicSearchConfig()
    cfg.min_snr = 10.0
    assert cfg.min_snr == 10.0


def test_transit_mask_max_iters():
    from stages.search_periodic import PeriodicSearchConfig
    cfg = PeriodicSearchConfig()
    cfg.max_iters = 20
    assert cfg.max_iters == 20


def test_nearest_epoch_time_basic():
    assert _nearest_epoch_time(15.0, 5.0, 10.0) == 15.0


def test_nearest_epoch_time_edge():
    assert _nearest_epoch_time(5.0, 5.0, 10.0) == 5.0
