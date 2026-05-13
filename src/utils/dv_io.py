import json
from pathlib import Path
from core.planet_candidate import PlanetCandidate
import glob


def load_candidates_for_target(target) -> list[PlanetCandidate]:
    target.load_state()
    rel = getattr(target, "last_candidates_run", None)
    if not rel:
        return []
    p = (target.root_dir / rel)
    if not p.exists():
        return []
    d = json.loads(p.read_text())

    # common shapes: either list under a key, or dict with "candidates"
    raw = d.get("candidates", None)
    if raw is None:
        # fallback: if the whole file is itself a list
        raw = d if isinstance(d, list) else []

    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(PlanetCandidate.from_dict(item))
    return out

def find_cached_tglc_aperture_fits(target, sector: int):
    pix_dir = Path(target.root_dir) / "pixels" / "TGLC"
    pats = [
        str(pix_dir / f"sector{int(sector):02d}_*.fits"),
        str(pix_dir / f"*s{int(sector):04d}*.fits"),  # if original names include s0007 style
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0])
    return None