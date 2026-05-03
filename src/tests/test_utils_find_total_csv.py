import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/bobby/dev/python/lpp_search/src/tests/parent")))

from utils.find_total_csv import find_total_csv


def test_exact_match(tmp_workdir):
    d = tmp_workdir / "target_tic-999"
    d.mkdir()
    (d / "target_tic-999_TGLC_APER__total.csv").touch()
    
    result = find_total_csv(d, "TGLC")
    assert result.exists()
    assert "TGLC" in str(result)


def test_fallback_to_any_total(tmp_workdir):
    d = tmp_workdir / "target_tic-999"
    d.mkdir()
    (d / "target_tic-999_TGLC_other_total.csv").touch()
    (d / "target_tic-999_SPOC_other_total.csv").touch()
    
    result = find_total_csv(d, "TGLC")
    assert result.exists()


def test_no_match_raises(tmp_workdir):
    d = tmp_workdir / "target_tic-999"
    d.mkdir()
    (d / "no_total_files.csv").touch()
    
    with pytest.raises(FileNotFoundError, match="No merged total"):
        find_total_csv(d, "TGLC")
