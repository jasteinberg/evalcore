"""scripts/preflight.py is a thin front end to core.calibrate: what is left
to test here is its argument handling and the wall-clock arithmetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from evalcore import HTTPBackend

PATH = Path(__file__).resolve().parents[2] / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight", PATH)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_wall_clock_arithmetic():
    # 1200 units x 0.5 s / 8 workers = 75 s; 8 workers / 0.5 s = 960 req/min
    minutes, rpm = preflight.wall_clock(1200, 0.5, 8)
    assert minutes == pytest.approx(75 / 60) and rpm == pytest.approx(960)


def test_model_is_required_and_never_defaulted():
    with pytest.raises(SystemExit):
        preflight.parse_args(["--provider", "openai"])
    assert preflight.parse_args(["--provider", "openai",
                                 "--list-models"]).model is None


def test_no_key_exits_2(monkeypatch):
    monkeypatch.delenv(HTTPBackend.ENV["openai"], raising=False)
    assert preflight.main(["--provider", "openai", "--model", "m"]) == 2
