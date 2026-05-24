"""Tests for utils/run_json.py (~100% coverage)."""
import pytest
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/bobby/dev/python/lpp_search/src/tests/parent")))

from utils.run_json import upsert_run_json, append_run_json_list


@pytest.fixture
def run_json_path(tmp_workdir):
    return tmp_workdir / "test_run.json"


def test_upsert_new_file(run_json_path):
    upsert_run_json(run_json_path, {"key1": "value1"})
    assert run_json_path.exists()
    data = json.loads(run_json_path.read_text())
    assert data == {"key1": "value1"}


def test_upsert_existing_file(run_json_path):
    run_json_path.write_text(json.dumps({"key1": "old"}))
    upsert_run_json(run_json_path, {"key2": "new"})
    
    data = json.loads(run_json_path.read_text())
    assert data["key1"] == "old"
    assert data["key2"] == "new"


def test_upsert_replaces_value(run_json_path):
    run_json_path.write_text(json.dumps({"key1": "old"}))
    upsert_run_json(run_json_path, {"key1": "new"})
    
    data = json.loads(run_json_path.read_text())
    assert data["key1"] == "new"


def test_append_new_list(run_json_path):
    append_run_json_list(run_json_path, "items", "a")
    
    data = json.loads(run_json_path.read_text())
    assert data["items"] == ["a"]


def test_append_to_existing_list(run_json_path):
    run_json_path.write_text(json.dumps({"items": ["a"]}))
    append_run_json_list(run_json_path, "items", "b")
    
    data = json.loads(run_json_path.read_text())
    assert data["items"] == ["a", "b"]


def test_append_nonexistent_list(run_json_path):
    append_run_json_list(run_json_path, "newkey", "first")
    
    data = json.loads(run_json_path.read_text())
    assert data["newkey"] == ["first"]


def test_append_replaces_nonlist(run_json_path):
    run_json_path.write_text(json.dumps({"items": "not_a_list"}))
    append_run_json_list(run_json_path, "items", "new")
    
    data = json.loads(run_json_path.read_text())
    assert data == {"items": ["new"]}
