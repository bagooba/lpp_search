# tests/test_dataprep.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

def test_match_logg_and_teff_for_LDC(fake_ldc_table, tmp_workdir):
    from stages.dataprep import match_logg_and_teff_for_LDC
    df = pd.DataFrame({
        "Teff": [3500, 3650, 3000],   # 3000 outside range, should still handle
        "logg": [5.0,   np.nan, 5.2], # one NaN logg triggers median in k-nearest pool
    })
    out = match_logg_and_teff_for_LDC(df.copy())
    assert "aLSM" in out.columns and "bLSM" in out.columns
    # Should fill finite a,b where possible
    assert np.isfinite(out["aLSM"]).iloc[0]
    assert np.isfinite(out["bLSM"]).iloc[0]

def test_get_catalog_info_reads_in_chunks(fake_mdwarf_catalog, fake_ldc_table, tmp_workdir):
    from stages.dataprep import get_catalog_info
    # Return rtrn_df to inspect
    df = get_catalog_info(123456789, rtrn_df=True)
    assert len(df) == 1
    assert df.iloc[0]["TICID"] == 123456789
    assert "aLSM" in df.columns and "bLSM" in df.columns

def test_extract_data_from_fits_files_writes_csv(stub_fits, tmp_workdir):
    from stages.dataprep import extract_data_from_fits_files
    # Create a fake FITS file path (content is stubbed by stub_fits)
    fits_path = tmp_workdir / "TIC123456789" / "fake-s13-lc.fits"
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    fits_path.write_text("placeholder")  # path presence only

    df = extract_data_from_fits_files(str(fits_path), PL="TGLC", sector=13)
    assert isinstance(df, pd.DataFrame)
    # Output file is named by parent dir basename and sector
    out_csv = fits_path.parent / "TGLC_TIC123456789_sector13.csv"
    assert out_csv.exists()

def test_get_data_flattens_and_merges(tmp_workdir, stub_fits, stub_flatten, stub_T14):
    from stages.dataprep import extract_data_from_fits_files, get_data
    # Prepare a directory with two sector CSVs via extraction
    d = tmp_workdir / "TIC123456789"
    d.mkdir(parents=True, exist_ok=True)
    for sector in (1, 2):
        fake_fits = d / f"fake-s{sector:02d}-lc.fits"
        fake_fits.write_text("placeholder")
        extract_data_from_fits_files(str(fake_fits), PL="TGLC", sector=sector)

    # Make a minimal catalog_df
    catalog_df = pd.DataFrame([{"Mass": 0.45, "Rad": 0.48}])
    out = get_data(str(d), flux_type="APER_", PL="TGLC", verbose=False, catalog_df=catalog_df)

    # The merged total should exist
    total_csv = d / "TIC123456789_TGLC_APER__total.csv"
    assert total_csv.exists()
    assert isinstance(out, pd.DataFrame)
    # Basic columns present
    assert set(["TIME", "RAW_FLUX", "FLUX", "FLUX_ERR", "FLUX_TREND"]).issubset(out.columns)

def test_DataPrep_prepare_orchestrates(tmp_workdir, TinyTarget, fake_mdwarf_catalog, fake_ldc_table,
                                       stub_fits, stub_flatten, stub_T14):
    # Arrange: build a target dir with two FITS files
    tic_dir = tmp_workdir / "TIC123456789"
    tic_dir.mkdir(parents=True, exist_ok=True)
    for sector in (10, 11):
        (tic_dir / f"fake-s{sector:02d}-lc.fits").write_text("placeholder")

    # Use the TinyTarget shim
    t = TinyTarget(ticid=123456789, root_dir=tic_dir)

    # Import after fixtures configured to capture patched symbols
    from stages.dataprep import DataPrep, PipelineStage

    dp = DataPrep(target=t, flavour="TGLC")
    total_path = dp.prepare()

    # Stage progression
    assert t.pipeline_stage == PipelineStage.MERGED
    # Ensure catalog was created and lifted
    cat = tic_dir / "tic_star_parameters.csv"
    assert cat.exists()
    # Total file written
    assert total_path.exists()