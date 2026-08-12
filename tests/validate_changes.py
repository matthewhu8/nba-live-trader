"""
Validation script for the four changes made before MMoE retrain.

Tests (synthetic data only — no MotherDuck required):
  A  forward_with_gates() outputs match forward() element-wise
  B  _gate_entropy() returns correct Shannon entropy values
  C  _compute_loss() with lambda_entropy > 0 reduces total loss
  D  Market scaler re-fit targets joint rows only (yes_bid mean in [30, 70])

Run from project root:
    python tests/validate_changes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import numpy as np
import pandas as pd
import torch

from models.mmoe.dataset import _build_tensor_dataset
from models.mmoe.feature_config import ALL_FEATURE_COLS, MARKET_COLS, PHYSICS_COLS, PREGAME_COLS
from models.mmoe.model import MMoEModel, N_EXPERTS
from models.mmoe.trainer import IDX_HAZ, IDX_HAS_RUN, IDX_MKT, IDX_RUN, IDX_TRAJ, IDX_X, _compute_loss, _gate_entropy

PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []  # (check_name, status, detail)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "✓" if condition else "✗"
    print(f"  {icon} {name}" + (f": {detail}" if detail else ""))


# ── A: forward_with_gates() correctness ──────────────────────────────────────

print("\n=== A: forward_with_gates() correctness ===")

torch.manual_seed(42)
model = MMoEModel()

# Equivalence must be checked in eval mode: Dropout is a no-op so both
# code paths (experts computed once vs. three times) are deterministic and
# identical. In train mode, each _mix_experts call samples a fresh dropout
# mask, so forward() and forward_with_gates() intentionally diverge —
# forward_with_gates() is the training-path replacement, not a copy.
model.eval()

B = 16
x = torch.randn(B, len(ALL_FEATURE_COLS))

with torch.no_grad():
    ha, hb, hc = model(x)
    ha2, hb2, hc2, ga, gb, gc = model.forward_with_gates(x)

check(
    "head_a_matches",
    torch.allclose(ha, ha2, atol=1e-6),
    f"max_diff={float((ha - ha2).abs().max()):.2e}",
)
check(
    "head_b_matches",
    torch.allclose(hb, hb2, atol=1e-6),
    f"max_diff={float((hb - hb2).abs().max()):.2e}",
)
check(
    "head_c_matches",
    torch.allclose(hc, hc2, atol=1e-6),
    f"max_diff={float((hc - hc2).abs().max()):.2e}",
)

for gate_name, g in [("gate_a", ga), ("gate_b", gb), ("gate_c", gc)]:
    check(
        f"{gate_name}_shape",
        g.shape == (B, N_EXPERTS),
        f"got {tuple(g.shape)}",
    )
    row_sums = g.sum(dim=-1)
    check(
        f"{gate_name}_rows_sum_to_1",
        bool(torch.allclose(row_sums, torch.ones(B), atol=1e-5)),
        f"max_dev={float((row_sums - 1).abs().max()):.2e}",
    )
    check(
        f"{gate_name}_values_in_01",
        bool((g > 0).all() and (g <= 1).all()),
        f"min={float(g.min()):.4f} max={float(g.max()):.4f}",
    )


# ── B: _gate_entropy() math ───────────────────────────────────────────────────

print("\n=== B: _gate_entropy() math ===")

uniform   = torch.tensor([[1/3, 1/3, 1/3]])
collapsed = torch.tensor([[0.994, 0.003, 0.003]])
two_exp   = torch.tensor([[0.5, 0.5, 0.0]])

H_uniform   = float(_gate_entropy(uniform))
H_collapsed = float(_gate_entropy(collapsed))
H_two       = float(_gate_entropy(two_exp))

check(
    "uniform_entropy_approx_ln3",
    abs(H_uniform - math.log(3)) < 0.001,
    f"H={H_uniform:.6f}, expected≈{math.log(3):.6f}",
)
check(
    "collapsed_entropy_near_zero",
    H_collapsed < 0.1,
    f"H={H_collapsed:.6f}",
)
check(
    "two_expert_entropy_approx_ln2",
    abs(H_two - math.log(2)) < 0.01,
    f"H={H_two:.6f}, expected≈{math.log(2):.6f}",
)
check(
    "all_entropies_nonnegative",
    H_uniform >= 0 and H_collapsed >= 0 and H_two >= 0,
    f"values: {H_uniform:.4f}, {H_collapsed:.4f}, {H_two:.4f}",
)


# ── C: _compute_loss() entropy term reduces loss ─────────────────────────────

print("\n=== C: _compute_loss() entropy term ===")

torch.manual_seed(7)
B = 32
x_loss = torch.randn(B, len(ALL_FEATURE_COLS))

model.eval()
with torch.no_grad():
    ha_l, hb_l, hc_l, ga_l, gb_l, gc_l = model.forward_with_gates(x_loss)

# Synthetic batch tensors
target_run  = torch.randint(0, 2, (B,)).float()
target_traj = torch.randn(B, 10) * 0.2
target_haz  = torch.rand(B, 10)
has_mkt     = torch.ones(B)
has_run     = torch.ones(B)
batch = [x_loss, target_run, target_traj, target_haz, has_mkt, has_run]

total_base, _, _, _ = _compute_loss(ha_l, hb_l, hc_l, batch, 1.0, 1.0, 0.3, lambda_entropy=0.0)
total_ent,  _, _, _ = _compute_loss(
    ha_l, hb_l, hc_l, batch, 1.0, 1.0, 0.3,
    gate_a=ga_l, gate_b=gb_l, gate_c=gc_l, lambda_entropy=0.02,
)

check(
    "entropy_reduces_loss",
    float(total_ent) < float(total_base),
    f"base={float(total_base):.6f} with_entropy={float(total_ent):.6f}",
)

# lambda=0 path must be identical whether gates are passed or not
total_no_lambda, _, _, _ = _compute_loss(
    ha_l, hb_l, hc_l, batch, 1.0, 1.0, 0.3,
    gate_a=ga_l, gate_b=gb_l, gate_c=gc_l, lambda_entropy=0.0,
)
check(
    "lambda0_identical_with_gates",
    float(total_base) == float(total_no_lambda),
    f"base={float(total_base):.8f} no_lambda={float(total_no_lambda):.8f}",
)


# ── D: Market scaler fix ──────────────────────────────────────────────────────

print("\n=== D: Market scaler fix (synthetic DataFrame) ===")

N = 5000
rng = np.random.default_rng(0)

# Build synthetic feature matrix — 58 cols, all zeros by default
df_data: dict[str, np.ndarray] = {col: np.zeros(N) for col in ALL_FEATURE_COLS}

# 6% rows are joint (has_market_data=1) — match live training distribution
joint_idx = rng.choice(N, size=int(N * 0.06), replace=False)
df_data["has_market_data"][joint_idx] = 1.0

# Fill joint market features with realistic live ranges
df_data["yes_bid"][joint_idx]                  = rng.uniform(30, 70, size=len(joint_idx))
df_data["yes_ask"][joint_idx]                  = df_data["yes_bid"][joint_idx] + rng.uniform(1, 4, len(joint_idx))
df_data["spread"][joint_idx]                   = rng.uniform(1, 5, len(joint_idx))
df_data["yes_last"][joint_idx]                 = rng.uniform(30, 70, len(joint_idx))
df_data["open_interest"][joint_idx]            = rng.uniform(100, 2000, len(joint_idx))
df_data["trade_volume_60s"][joint_idx]         = rng.uniform(0, 500, len(joint_idx))
df_data["time_since_last_trade_ms"][joint_idx] = rng.uniform(0, 60000, len(joint_idx))
df_data["open_interest_change_60s"][joint_idx] = rng.uniform(-50, 50, len(joint_idx))
df_data["d_yes_bid"][joint_idx]                = rng.uniform(-5, 5, len(joint_idx))
df_data["d_spread"][joint_idx]                 = rng.uniform(-2, 2, len(joint_idx))
df_data["bid_velocity_30s"][joint_idx]         = rng.uniform(-0.5, 0.5, len(joint_idx))
df_data["bid_acceleration_30s"][joint_idx]     = rng.uniform(-0.1, 0.1, len(joint_idx))
df_data["bid_vs_last_divergence"][joint_idx]   = rng.uniform(-3, 3, len(joint_idx))

# Minimal target columns so _build_tensor_dataset doesn't error
df_data["target_meaningful_run_5_scoring"] = rng.integers(0, 2, N).astype(float)
for i in range(10):
    df_data[f"traj_{i}"] = rng.uniform(-0.3, 0.3, N)
    df_data[f"haz_{i}"]  = rng.uniform(0, 1, N)

df = pd.DataFrame(df_data)

dataset, scaler = _build_tensor_dataset(df, fit_scaler=True)

yes_bid_idx       = ALL_FEATURE_COLS.index("yes_bid")        # 69
has_market_idx    = ALL_FEATURE_COLS.index("has_market_data") # 82

bid_mean  = float(scaler.mean_[yes_bid_idx])
bid_scale = float(scaler.scale_[yes_bid_idx])
mkt_mean  = float(scaler.mean_[has_market_idx])

check(
    "yes_bid_mean_in_joint_range",
    30 <= bid_mean <= 70,
    f"yes_bid mean={bid_mean:.2f}¢ (expected 30–70, old global was ~3)",
)
check(
    "yes_bid_scale_reasonable",
    bid_scale > 5,
    f"yes_bid std={bid_scale:.2f}",
)
check(
    "has_market_data_mean_unchanged",
    mkt_mean < 0.2,
    f"has_market_data mean={mkt_mean:.4f} (should reflect ~6% base rate)",
)

X_tensor = dataset.tensors[IDX_X]
X_np = X_tensor.numpy()
n_nonfinite = int((~np.isfinite(X_np)).sum())
check(
    "no_nan_inf_in_X_scaled",
    n_nonfinite == 0,
    f"non-finite count={n_nonfinite}",
)


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)

for name, status, detail in results:
    icon = "✓" if status == PASS else "✗"
    line = f"  {icon} {name}"
    if status == FAIL:
        line += f"  ← FAIL ({detail})"
    print(line)

print(f"\n{passed}/{passed + failed} checks passed")

if failed:
    print("\nFAILED — do not retrain until all checks pass")
    sys.exit(1)
else:
    print("\nAll checks PASSED — safe to retrain")
