"""
MMoE Inference Wrapper.

Thin wrapper that loads the trained MMoE model + scaler and provides a clean
predict() interface matching the existing RunPredictor / KalshiPriceMovementPredictor
pattern.

Usage:
    predictor = MMoEPredictor.load()
    output = predictor.predict(feature_dict, yes_bid=52, yes_ask=55)
    print(output.run_prob)         # float in [0, 1]
    print(output.trajectory)       # list[float], 10 log-odds deltas
    print(output.hazard)           # list[float], 10 survival hazards in [0, 1]
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from models.mmoe.feature_config import ALL_FEATURE_COLS
from models.mmoe.model import MMoEModel

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_PATH  = ROOT_DIR / "models/saved/mmoe_delay20.pt"
SCALER_PATH = ROOT_DIR / "models/saved/mmoe_scaler_delay20.pkl"


@dataclass
class MMoEOutput:
    """
    Inference output for a single possession.

    The first three fields (run_prob, trajectory, hazard) are the gated
    head outputs — what the model actually believes after combining experts.
    These are sufficient for the trading decision.

    The remaining four fields (added in Phase 5) are interpretability
    diagnostics for "what is the model thinking" post-mortems:

      - gating_weights: 3×3 matrix [head][expert] of softmax weights —
        which experts each head trusted for this input.
      - expert_opinions_*: per-expert predictions for each head — what
        the head WOULD output if it trusted only one expert. Reveals
        when experts disagree (borderline regimes) vs agree (consensus).

    All diagnostic fields default to None so existing callers (predict_batch
    in particular) keep working without change.
    """
    run_prob:   float
    trajectory: list[float]
    hazard:     list[float]
    # Phase 5 interpretability diagnostics (None unless predict() was used)
    gating_weights:        Optional[list[list[float]]] = None  # 3 heads × 3 experts
    expert_opinions_run:   Optional[list[float]] = None        # 3 floats (one per expert)
    expert_opinions_traj:  Optional[list[list[float]]] = None  # 3 × 10
    expert_opinions_haz:   Optional[list[list[float]]] = None  # 3 × 10


class MMoEPredictor:
    """
    Inference wrapper for the trained MMoE model.

    Handles feature alignment, scaling, and device placement automatically.
    All three head outputs are returned on every predict() call.
    """

    def __init__(self, model: MMoEModel, scaler: StandardScaler) -> None:
        self._model  = model
        self._scaler = scaler
        self._device = torch.device("cpu")
        self._model.eval()

    @classmethod
    def load(
        cls,
        model_path:  Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
    ) -> "MMoEPredictor":
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Run `python -m models.mmoe.train_mmoe` first."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Run `python -m models.mmoe.train_mmoe` first."
            )

        model = MMoEModel()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])

        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        return cls(model, scaler)

    def predict(
        self,
        feature_dict: dict,
        yes_bid: Optional[float] = None,
        yes_ask: Optional[float] = None,
    ) -> MMoEOutput:
        """
        Run inference on a single possession's features.

        Args:
            feature_dict: dict mapping feature names to scalar values.
                          Missing features default to 0.0.
                          X_market features (yes_bid, yes_ask, etc.) can be
                          passed here or via the convenience kwargs below.
            yes_bid:      Current best bid in cents (fills feature_dict["yes_bid"])
            yes_ask:      Current best ask in cents (fills feature_dict["yes_ask"])

        Returns:
            MMoEOutput with run_prob, trajectory, hazard.
        """
        fd = dict(feature_dict)
        if yes_bid is not None:
            fd["yes_bid"] = float(yes_bid)
            fd.setdefault("has_market_data", 1.0)
        if yes_ask is not None:
            fd["yes_ask"] = float(yes_ask)
            fd.setdefault("spread", float(yes_ask) - float(fd.get("yes_bid", yes_ask)))

        # Align to ALL_FEATURE_COLS order; fill missing with 0
        x_raw = np.array(
            [float(fd.get(col, 0.0)) for col in ALL_FEATURE_COLS],
            dtype=np.float32,
        ).reshape(1, -1)

        x_scaled = self._scaler.transform(x_raw).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled).to(self._device)

        # predict_with_diagnostics returns gated outputs (mathematically
        # identical to forward()) plus gating weights and per-expert
        # opinions. Computing experts once and reusing them is actually
        # faster than the forward() path for inference.
        with torch.no_grad():
            diag = self._model.predict_with_diagnostics(x_tensor)

        head_a, head_b, head_c = diag["gated"]
        gate_a, gate_b, gate_c = diag["gates"]
        op_a,   op_b,   op_c   = diag["opinions"]

        # Squeeze the batch dimension (B=1 for single-row inference) to
        # keep JSON-friendly shapes:
        #   gating_weights: 3 × 3 (head × expert)
        #   expert_opinions_run:  3 floats
        #   expert_opinions_traj: 3 × 10
        #   expert_opinions_haz:  3 × 10
        return MMoEOutput(
            run_prob   = float(head_a.squeeze().item()),
            trajectory = head_b.squeeze().tolist(),
            hazard     = head_c.squeeze().tolist(),
            gating_weights = [
                gate_a.squeeze(0).tolist(),
                gate_b.squeeze(0).tolist(),
                gate_c.squeeze(0).tolist(),
            ],
            expert_opinions_run  = op_a.squeeze(0).squeeze(-1).tolist(),
            expert_opinions_traj = op_b.squeeze(0).tolist(),
            expert_opinions_haz  = op_c.squeeze(0).tolist(),
        )

    def get_feature_stats(self) -> dict[str, tuple[float, float]]:
        """
        Return per-feature (mean, std) from the trained scaler, keyed by
        feature name. Used by the inference service to compute z-scores
        against the training distribution for selected features (Phase 6
        interpretability — answers "is this feature value unusual?").

        The StandardScaler computes per-column statistics independently,
        so pulling out the i-th mean/std for the i-th feature name in
        ALL_FEATURE_COLS is exactly the per-feature stat that fitted on
        the training set.
        """
        means  = self._scaler.mean_
        scales = self._scaler.scale_
        return {
            name: (float(means[i]), float(scales[i]))
            for i, name in enumerate(ALL_FEATURE_COLS)
        }

    def predict_batch(self, feature_matrix: np.ndarray) -> list[MMoEOutput]:
        """
        Run inference on a batch of feature rows.

        Args:
            feature_matrix: (N, 83) float array, columns aligned to ALL_FEATURE_COLS.

        Returns:
            List of N MMoEOutput objects.
        """
        x_scaled = self._scaler.transform(feature_matrix).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled).to(self._device)

        with torch.no_grad():
            head_a, head_b, head_c = self._model(x_tensor)

        return [
            MMoEOutput(
                run_prob   = float(head_a[i].item()),
                trajectory = head_b[i].tolist(),
                hazard     = head_c[i].tolist(),
            )
            for i in range(len(feature_matrix))
        ]
