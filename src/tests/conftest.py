# tests/conftest.py
import os
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """A clean temp directory for each test."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)

@pytest.fixture
def fake_ldc_table(monkeypatch):
    """Provide a small LDC table via config.LDC_PARAMS_MDWARF."""
    import importlib
    config = importlib.import_module("config")
    ldc = pd.DataFrame({
        "Teff": [3200, 3400, 3600, 3800],
        "logg": [4.8, 4.9, 5.0, 5.1],
        "aLSM": [0.35, 0.33, 0.31, 0.30],
        "bLSM": [0.25, 0.27, 0.29, 0.30],
    })
    monkeypatch.setattr(config, "LDC_PARAMS_MDWARF", ldc, raising=False)
    return ldc

@pytest.fixture
def fake_mdwarf_catalog(monkeypatch, tmp_workdir):
    """Provide config.MDWARF_CATALOG path and file with TIC row."""
    import importlib
    config = importlib.import_module("config")
    catalog_path = tmp_workdir / "mdwarf_catalog.csv"
    df = pd.DataFrame({
        "TICID": [123456789],
        "Teff": [3500],
        "logg": [5.0],
        "Mass": [0.45],
        "eMass": [0.05],
        "Rad": [0.48],
        "eRad": [0.04],
    })
    df.to_csv(catalog_path, index=False)
    monkeypatch.setattr(config, "MDWARF_CATALOG", str(catalog_path), raising=False)
    return catalog_path

@pytest.fixture
def stub_T14(monkeypatch):
    """
    Stub transit duration function so window length is predictable.
    """
    import importlib
    dp = importlib.import_module("stages.dataprep")
    def _T14(P, R_star, M_star, R_planet):
        # Return something deterministic in days; keep scale ~hours
        #  e.g., ~3 hours = 0.125 day for default values
        return 0.125
    monkeypatch.setattr(dp, "T14", _T14, raising=True)
    return _T14

@pytest.fixture
def stub_flatten(monkeypatch):
    """
    Stub flatten(time, flux, method, window_length, return_trend=True)
    Returns (flat_flux, trend) where flat=flux/median(trend)=~1 with small noise.
    """
    import importlib
    dp = importlib.import_module("stages.dataprep")

    def _flatten(time, flux, method=None, window_length=None, return_trend=True, **kwargs):
        # Simple moving median trend around 1.0 for predictable outputs
        trend = np.full_like(flux, np.nanmedian(flux))
        flat = flux / (trend + 1e-12)
        return (flat, trend) if return_trend else flat

    monkeypatch.setattr(dp, "flatten", _flatten, raising=True)
    return _flatten

@pytest.fixture
def stub_fits(monkeypatch):
    """
    Stub astropy.io.fits.open to yield a fake HDU with a table-like data object.
    We'll synthesize columns needed by extract_data_from_fits_files.
    """
    class FakeColumn:
        def __init__(self, name): self.name = name

    class FakeTable:
        def __init__(self, data, cols):
