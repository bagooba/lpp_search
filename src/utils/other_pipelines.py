# utils/other_pipelines.py
#
# Purpose:
#   Fetch other pipeline light curve files (QLP, TESS-SPOC, SPOC) via lightkurve,
#   extract a consistent TIME/FLUX(/FLUX_ERR/...) table using your existing
#   extract_data_from_fits_files(), and store a single stitched Parquet per
#   (target, pipeline) for phase-folded DV plotting.
#
# Design goals:
#   - Small functions that compose.
#   - "Get if already there, else download+extract+save" behavior.
#   - Produce ONE stitched parquet per pipeline per target (across all sectors)
#     so phase-folding works without per-sector bookkeeping.
#
# Pipelines supported:
#   - QLP
#   - TESS-SPOC
#   - SPOC_2min  (author='SPOC', exptime='short')
#   - SPOC_30min (author='SPOC', exptime='long')
#
# Notes:
#   - This module avoids any dependence on target "total.csv" products.
#   - Disk minimization: you can set delete_fits=True to remove downloaded FITS
#     after converting to Parquet.

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import lightkurve as lk
from astropy.io import fits as apf

from stages.dataprep import extract_data_from_fits_files


# -----------------------------
# Small, composable utilities
# -----------------------------

def lk_query_for_pipeline(pipeline: str) -> dict:
    """Return keyword args for lk.search_lightcurve() for a given pipeline label."""
    if pipeline == "QLP":
        return {"author": "QLP"}
    if pipeline == "TESS-SPOC":
        return {"author": "TESS-SPOC"}
    if pipeline == "SPOC_2min":
        return {"author": "SPOC", "exptime": "short"}
    if pipeline == "SPOC_30min":
        return {"author": "SPOC", "exptime": "long"}
    raise ValueError(f"Unknown pipeline '{pipeline}'")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def stitched_parquet_path(target, pipeline: str, out_dir: Optional[Path] = None) -> Path:
    """Where to store the stitched parquet for (target, pipeline)."""
    base = Path(out_dir) if out_dir is not None else (Path(target.root_dir) / "external" / "parquet_lc")
    ensure_dir(base)
    return base / f"{pipeline}_TIC{int(target.ticid)}.parquet"


def download_dir_path(target, download_dir: Optional[Path] = None) -> Path:
    base = Path(download_dir) if download_dir is not None else (Path(target.root_dir) / "external" / "mast_lc")
    ensure_dir(base)
    return base


def sector_from_header_or_name(fits_path: str) -> int:
    """Best-effort sector from FITS header or s#### token in filename."""
    try:
        with apf.open(fits_path, memmap=True) as hdul:
            hdr = hdul[0].header
            if "SECTOR" in hdr:
                return int(hdr["SECTOR"])
    except Exception:
        pass
    m = re.search(r"s(\d{4})", str(fits_path))
    return int(m.group(1)) if m else 0


# -----------------------------
# Parquet IO (small)
# -----------------------------

def save_lc_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Store a compact Parquet containing only the DV-needed columns."""
    keep = [c for c in ["TIME", "FLUX", "FLUX_ERR", "BKG_FLUX", "CENTROID_X", "CENTROID_Y", "QUALITY"] if c in df.columns]
    df2 = df[keep].copy()

    # quality filtering early reduces size and helps plots
    if "QUALITY" in df2.columns:
        df2 = df2[df2["QUALITY"] == 0]

    # downcast
    for c in df2.columns:
        if c == "QUALITY":
            df2[c] = df2[c].astype("int32", copy=False)
        else:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("float32", copy=False)

    df2.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)


def read_stitched_parquet(parq_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, flux, err arrays for plotting."""
    df = pd.read_parquet(parq_path, engine="pyarrow")
    if "TIME" not in df.columns or "FLUX" not in df.columns:
        raise KeyError(f"Missing TIME/FLUX in parquet: {parq_path}")

    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)

    if "FLUX_ERR" in df.columns:
        err = df["FLUX_ERR"].to_numpy(float)
        if len(err) != len(flux):
            err = np.full(len(flux), np.nanstd(flux), dtype=float)
    else:
        err = np.full(len(flux), np.nanstd(flux), dtype=float)


    # ensure monotonic time
    if not np.all(np.diff(time) >= 0):
        idx = np.argsort(time)
        outlst = [time[idx], flux[idx], err[idx]]
        if 'CENTROID_X' and 'CENTROID_Y' in df.columns:
            if (not np.any(np.abs(df['CENTROID_X'])>=0)) and (not np.any(np.abs(df['CENTROID_Y'])>=0)):
                outlst += [df['CENTROID_X'].to_numpy(float)[idx], df['CENTROID_Y'].to_numpy(float)[idx]]

        

    
    return outlst


# -----------------------------
# Internet download (small)
# -----------------------------

def download_fits_for_pipeline(
    target,
    pipeline: str,
    *,
    download_dir: Path,
    sector: Optional[int] = None,
) -> List[str]:
    """Search & download FITS LCs for one pipeline. Returns local FITS paths."""
    q = lk_query_for_pipeline(pipeline)
    sr = lk.search_lightcurve(
        f"TIC {int(target.ticid)}",
        mission="TESS",
        sector=sector,
        **q,
    )

    if sr is None or len(sr) == 0:
        return []

    downloaded = sr.download_all(download_dir=str(download_dir))

    # normalize into file paths
    paths: List[str] = []
    try:
        items = list(downloaded)
    except Exception:
        items = [downloaded]

    for obj in items:
        if obj is None:
            continue
        if isinstance(obj, (str, Path)):
            p = str(obj)
        else:
            fn = getattr(obj, "filename", None)
            p = str(fn) if fn else None
        if p and p.endswith(".fits"):
            paths.append(p)

    return sorted(set(paths))


# -----------------------------
# FITS -> extracted DF -> stitched parquet (small-ish)
# -----------------------------

def extract_many_fits_to_df(
    fits_paths: List[str],
    *,
    pipeline: str,
    sector: Optional[int] = None,
    delete_fits: bool = True,
) -> pd.DataFrame:
    """Run your extract_data_from_fits_files on each FITS and concat the results."""
    dfs: List[pd.DataFrame] = []

    for fp in fits_paths:
        sec = sector_from_header_or_name(fp)
        if sector is not None and int(sec) != int(sector):
            continue

        df = extract_data_from_fits_files(fp, PL=pipeline, sector=sec)
        if df is not None and len(df) > 0:
            dfs.append(df)

        if delete_fits:
            try:
                Path(fp).unlink()
            except Exception:
                pass

    if not dfs:
        raise RuntimeError(f"No extracted data produced for pipeline={pipeline}")

    df_all = pd.concat(dfs, ignore_index=True)

    if "QUALITY" in df_all.columns:
        df_all = df_all[df_all["QUALITY"] == 0]

    # Ensure canonical FLUX (your extractor may emit multiple *_FLUX columns)
    if "FLUX" not in df_all.columns:
        flux_candidates = [c for c in df_all.columns if ("FLUX" in c.upper() and "ERR" not in c.upper())]
        if not flux_candidates:
            raise KeyError("No FLUX-like column after extraction")
        flux_col = sorted(flux_candidates, key=len)[0]
        df_all = df_all.rename(columns={flux_col: "FLUX"})
        if flux_col + "_ERR" in df_all.columns:
            df_all = df_all.rename(columns={flux_col + "_ERR": "FLUX_ERR"})

    df_all = df_all.sort_values("TIME").reset_index(drop=True)
    return df_all


def build_stitched_parquet(
    target,
    pipeline: str,
    *,
    out_dir: Optional[Path] = None,
    download_dir: Optional[Path] = None,
    delete_fits: bool = True,
    sector: Optional[int] = None,
) -> Path:
    """Download (if needed), extract, stitch, and save one parquet for this pipeline."""
    parq = stitched_parquet_path(target, pipeline, out_dir=out_dir)

    # download
    dl_dir = download_dir_path(target, download_dir=download_dir)
    fits_paths = download_fits_for_pipeline(target, pipeline, download_dir=dl_dir, sector=sector)
    if not fits_paths:
        raise FileNotFoundError(
            f"No FITS products found for TIC {int(target.ticid)} pipeline={pipeline} sector={sector}"
        )

    # extract + stitch
    df_all = extract_many_fits_to_df(fits_paths, pipeline=pipeline, sector=sector, delete_fits=delete_fits)

    # save
    save_lc_parquet(df_all, parq)
    return parq


def get_or_build_stitched_parquet(
    target,
    pipeline: str,
    *,
    out_dir: Optional[Path] = None,
    download_dir: Optional[Path] = None,
    delete_fits: bool = True,
    sector: Optional[int] = None,
) -> Path:
    """Return stitched parquet if present; otherwise build it."""
    parq = stitched_parquet_path(target, pipeline, out_dir=out_dir)
    if parq.exists():
        return parq
    return build_stitched_parquet(
        target,
        pipeline,
        out_dir=out_dir,
        download_dir=download_dir,
        delete_fits=delete_fits,
        sector=sector,
    )


def get_time_flux_err_for_pipeline(
    target,
    pipeline: str,
    *,
    out_dir: Optional[Path] = None,
    download_dir: Optional[Path] = None,
    delete_fits: bool = True,
    sector: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: ensure stitched parquet exists, then return arrays for plotting."""
    parq = get_or_build_stitched_parquet(
        target,
        pipeline,
        out_dir=out_dir,
        download_dir=download_dir,
        delete_fits=delete_fits,
        sector=sector,
    )
