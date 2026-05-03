import pytest
from pathlib import Path
import sys
import os
sys.path.insert(0, str(Path("/Users/bobby/dev/python/lpp_search/src/tests/parent")))

from utils.queue import enqueue, QUEUE_ROOT


def test_enqueue_creates_marker(tmp_workdir, monkeypatch):
    monkeypatch.setattr("utils.queue.QUEUE_ROOT", tmp_workdir / "queue")
    
    result = enqueue("01_prepare", 123)
    assert result is None
    
    marker = tmp_workdir / "queue" / "01_prepare" / "123"
    assert marker.exists() is True


def test_enqueue_skips_existing(tmp_workdir, monkeypatch):
    monkeypatch.setattr("utils.queue.QUEUE_ROOT", tmp_workdir / "queue")
    
    # First create marker directory
    (tmp_workdir / "queue" / "01_prepare").mkdir(parents=True)
    (tmp_workdir / "queue" / "01_prepare" / "123").touch()
    
    # Try to enqueue again - should silently skip
    result = enqueue("01_prepare", 123)
    assert result is None
    
    # Only one marker should exist
    markers = list((tmp_workdir / "queue" / "01_prepare").glob("*"))
    assert len(markers) == 1


def test_enqueue_creates_dir(tmp_workdir, monkeypatch):
    monkeypatch.setattr("utils.queue.QUEUE_ROOT", tmp_workdir / "queue")
    
    # Queue root doesn't exist
    result = enqueue("02_process", 456)
    assert result is None
    
    # Directory and marker should be created
    assert (tmp_workdir / "queue" / "02_process" / "456").exists()
