from __future__ import annotations
import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.io import fits



def extract_tglc_aperture_stamp(
    fits_path: str | Path,
    *,
    epoch_index: int | None = None,
    epoch_time: float | None = None,
    stamp_size: int = 13,
    combine: str = "median",  # "median" or "mean"
) -> np.ndarray:
    """
    Extract a (stamp_size x stamp_size) image stamp from a TGLC 'aperture' FITS cutout.

    Assumptions (from your header/comment):
      - hdul[0].data has shape (nt, ny, nx)
      - STAR_X / STAR_Y give target position in cutout coordinates
      - TIME may be in an extension table; if not found, you can use epoch_index.

    If epoch_time is provided and TIME array exists, selects nearest cadence to that time.
    Otherwise uses epoch_index (default: middle cadence).
    If combine is "median"/"mean", returns a collapsed stamp over a small time window
    centered on the chosen cadence (see window_half below).
    """
    fits_path = Path(fits_path)

    with fits.open(fits_path, memmap=True) as hdul:
        cube = hdul[0].data
        hdr = hdul[0].header

        if cube is None or cube.ndim != 3:
            raise ValueError(f"Expected hdul[0].data to be 3D (time,y,x). Got {None if cube is None else cube.ndim}D")

        nt, ny, nx = cube.shape

        # star position (round to nearest pixel)
        sx = int(np.rint(float(hdr.get("STAR_X"))))
        sy = int(np.rint(float(hdr.get("STAR_Y"))))

        # pick cadence index
        k = None
        time_arr = None

        # try to find TIME in any table extension
        for h in hdul[1:]:
            if hasattr(h, "data") and h.data is not None and getattr(h.data, "names", None):
                names = [n.upper() for n in h.data.names]
                if "TIME" in names:
                    time_arr = np.array(h.data["TIME"], dtype=float)
                    break

        if epoch_time is not None and time_arr is not None and len(time_arr) == nt:
            k = int(np.nanargmin(np.abs(time_arr - float(epoch_time))))
        elif epoch_index is not None:
            k = int(epoch_index)
        else:
            k = nt // 2  # default: middle cadence

        k = max(0, min(nt - 1, k))

        # choose a small window of cadences around k for robustness
        # (you can change this later; this is just to make stamps stable)
        window_half = 2  # ±2 cadences
        i0 = max(0, k - window_half)
        i1 = min(nt, k + window_half + 1)

        # stamp bounds
        half = stamp_size // 2
        y0 = max(0, sy - half); y1 = min(ny, sy + half + 1)
        x0 = max(0, sx - half); x1 = min(nx, sx + half + 1)

        subcube = cube[i0:i1, y0:y1, x0:x1].astype(float)

        # collapse time dimension
        if combine == "mean":
            stamp = np.nanmean(subcube, axis=0)
        else:
            stamp = np.nanmedian(subcube, axis=0)

        # normalize for display (optional but helps look consistent)
        stamp = stamp - np.nanmin(stamp)
        mx = np.nanmax(stamp)
        if np.isfinite(mx) and mx > 0:
            stamp = stamp / mx

        return stamp



def make_pixel_stamp_from_tglc_aperture_fits(
    fits_path: str | Path,
    *,
    epoch_time: float | None = None,      # time in same units as TIME column if present (often TJD or BJD-2457000)
    epoch_index: int | None = None,       # fallback if no TIME column
    stamp_size: int = 13,                 # Eleanor default is 13x13 for its TPF cutout size 
    half_window_cadences: int = 0,        # 0 = single cadence; 2 = median over 5 cadences
    combine: str = "median",              # "median" or "mean"
    normalize: bool = True,               # normalize to [0,1] for display
) -> np.ndarray:
    """
    Extract a stamp centered on STAR_X/STAR_Y from a TGLC aperture cutout cube.

    Expected:
      - hdul[0].data has shape (nt, ny, nx)  [time, y, x]
      - header has STAR_X, STAR_Y
      - optional TIME column in a table extension (we search for it)

    Returns:
      stamp: 2D numpy array (stamp_size x stamp_size), clipped at edges if near border.
    """
    fits_path = Path(fits_path)

    with fits.open(fits_path, memmap=True) as hdul:
        cube = hdul[0].data
        hdr = hdul[0].header

        if cube is None or getattr(cube, "ndim", None) != 3:
            raise ValueError(f"Expected hdul[0].data to be 3D (time,y,x). Got: {None if cube is None else cube.ndim}D")

        nt, ny, nx = cube.shape

        # center from header (rounded to nearest pixel)
        sx = int(np.rint(float(hdr.get("STAR_X"))))
        sy = int(np.rint(float(hdr.get("STAR_Y"))))

        # try to find TIME array in any table extension
        time_arr = None
        for h in hdul[1:]:
            data = getattr(h, "data", None)
            names = getattr(data, "names", None)
            if data is not None and names is not None:
                up = [n.upper() for n in names]
                if "TIME" in up:
                    time_arr = np.array(data["TIME"], dtype=float)
                    break

        # choose cadence index k
        if epoch_time is not None and time_arr is not None and len(time_arr) == nt:
            k = int(np.nanargmin(np.abs(time_arr - float(epoch_time))))
        elif epoch_index is not None:
            k = int(epoch_index)
        else:
            k = nt // 2  # stable default

        k = max(0, min(nt - 1, k))

        # choose cadence window [k-half, k+half]
        i0 = max(0, k - int(half_window_cadences))
        i1 = min(nt, k + int(half_window_cadences) + 1)

        # stamp bounds around star
        half = stamp_size // 2
        y0 = max(0, sy - half)
        y1 = min(ny, sy + half + 1)
        x0 = max(0, sx - half)
        x1 = min(nx, sx + half + 1)

        subcube = np.array(cube[i0:i1, y0:y1, x0:x1], dtype=float)

        if combine == "mean":
            stamp = np.nanmean(subcube, axis=0)
        else:
            stamp = np.nanmedian(subcube, axis=0)

        if normalize:
            stamp = stamp - np.nanmin(stamp)
            mx = np.nanmax(stamp)
            if np.isfinite(mx) and mx > 0:
                stamp = stamp / mx

        return stamp
    

def plot_pixel_stamp(ax, stamp: np.ndarray | None, title: str):
    if stamp is None:
        ax.text(0.5, 0.5, "pixel stamp\nnot available", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return
    ax.imshow(stamp, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pixel_grid(ax, stamp):
    if stamp is None:
        ax.text(0.5, 0.5, "pixel grid\nnot available", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return
    ax.imshow(stamp, origin="lower", cmap="viridis")
    ax.set_axis_off()
