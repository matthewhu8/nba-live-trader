"""
Tests for the config-driven model-path override.

Model/scaler paths moved from hardcoded predictor.py constants into trading.yaml,
forwarded on /game/start. Python reloads only when the requested path differs
from what's loaded. These tests pin path resolution, the reload decision, and
the load helper — using a stub predictor so no real .pt is touched.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "live-trader"))

import inference.main as m
from models.mmoe.predictor import MODEL_PATH, ROOT_DIR


class _FakePredictor:
    """Stand-in for MMoEPredictor — only get_feature_stats() is exercised."""

    def get_feature_stats(self):
        return {name: (0.0, 1.0) for name in m._ZSCORE_FEATURES}


def _stub_loader(monkeypatch):
    """Patch MMoEPredictor.load to return a fake (no file I/O), and reset the
    module's loaded-path bookkeeping so each test starts clean."""
    monkeypatch.setattr(m.MMoEPredictor, "load", lambda mp, sp: _FakePredictor())
    m._predictor = None
    m._zscore_stats = {}
    m._loaded_model_path = None
    m._loaded_scaler_path = None


def test_resolve_path_relative_anchors_at_root():
    resolved = m._resolve_path("models/saved/x.pt")
    assert resolved == ROOT_DIR / "models/saved/x.pt"


def test_resolve_path_absolute_unchanged():
    abs_path = "/tmp/some/abs.pt"
    assert m._resolve_path(abs_path) == Path(abs_path)


def test_requested_paths_default_to_packaged_model():
    want_model, _ = m._requested_paths(None, None)
    assert want_model == str(MODEL_PATH)


def test_load_predictor_sets_globals(monkeypatch):
    _stub_loader(monkeypatch)
    assert m._load_predictor() is True
    assert m._predictor is not None
    assert m._loaded_model_path == str(MODEL_PATH)
    # z-score stats rebuilt from the (stub) scaler.
    assert len(m._zscore_stats) == len(m._ZSCORE_FEATURES)


def test_needs_reload_only_on_change(monkeypatch):
    _stub_loader(monkeypatch)
    m._load_predictor()  # loads the default model
    # Same path → no reload needed.
    assert m._needs_reload(None, None) is False
    assert m._needs_reload("models/saved/mmoe_delay20.pt", None) is False
    # Different path → reload needed.
    assert m._needs_reload("models/saved/other.pt", None) is True


def test_reload_swaps_loaded_path(monkeypatch):
    _stub_loader(monkeypatch)
    m._load_predictor()
    assert m._needs_reload("models/saved/other.pt", "models/saved/other.pkl") is True
    m._load_predictor("models/saved/other.pt", "models/saved/other.pkl")
    assert m._loaded_model_path == str(ROOT_DIR / "models/saved/other.pt")
    assert m._needs_reload("models/saved/other.pt", "models/saved/other.pkl") is False


def test_load_predictor_missing_artifact_returns_false(monkeypatch):
    _stub_loader(monkeypatch)

    def _raise(mp, sp):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(m.MMoEPredictor, "load", _raise)
    assert m._load_predictor() is False
    assert m._predictor is None
