"""
MMoE (Multi-gate Mixture-of-Experts) model.

Architecture:
  - 3 shared experts (Linear 83→64→64, BN + ReLU + Dropout)
  - 3 per-task gates (Linear 83→3, Softmax) — one per head
  - Head A: P(run) binary classifier  [sigmoid output]
  - Head B: 10-checkpoint price trajectory [raw output, log-odds delta units]
  - Head C: 10-horizon discrete survival hazard [sigmoid output per horizon]

Total parameters: ~37K (includes BatchNorm) — well-matched to the 24K joint rows in the training set.
"""

import torch
import torch.nn as nn
from torch import Tensor

from models.mmoe.feature_config import ALL_FEATURE_COLS

INPUT_DIM    = len(ALL_FEATURE_COLS)   # 83
EXPERT_DIM   = 64
N_EXPERTS    = 3
N_TRAJ       = 10
N_HAZARD     = 10
HEAD_HIDDEN  = 32


class Expert(nn.Module):
    """Single expert: two-layer MLP with BN, ReLU, Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Gate(nn.Module):
    """Per-task gating network: softmax over N_EXPERTS."""

    def __init__(self, input_dim: int, n_experts: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, n_experts)

    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(self.fc(x), dim=-1)  # (B, n_experts)


class TaskHead(nn.Module):
    """Shared head structure: Linear(expert_dim → hidden) → ReLU → Dropout → Linear(hidden → out)."""

    def __init__(
        self,
        expert_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(expert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MMoEModel(nn.Module):
    """
    Multi-gate Mixture-of-Experts model.

    Forward returns (head_a, head_b, head_c):
      head_a: (B, 1)  — P(run), sigmoid applied
      head_b: (B, 10) — trajectory log-odds deltas, no activation
      head_c: (B, 10) — hazard probs per horizon, sigmoid applied
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        expert_dim:  int   = EXPERT_DIM,
        n_experts:   int   = N_EXPERTS,
        head_hidden: int   = HEAD_HIDDEN,
        expert_dropout: float = 0.3,
        head_dropout:   float = 0.2,
    ) -> None:
        super().__init__()

        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim, dropout=expert_dropout)
            for _ in range(n_experts)
        ])

        self.gate_a = Gate(input_dim, n_experts)
        self.gate_b = Gate(input_dim, n_experts)
        self.gate_c = Gate(input_dim, n_experts)

        self.head_a = TaskHead(expert_dim, head_hidden, 1,        dropout=head_dropout, use_sigmoid=True)
        self.head_b = TaskHead(expert_dim, head_hidden, N_TRAJ,   dropout=head_dropout, use_sigmoid=False)
        self.head_c = TaskHead(expert_dim, head_hidden, N_HAZARD, dropout=head_dropout, use_sigmoid=True)

    def _mix_experts(self, x: Tensor, gate: Gate) -> Tensor:
        """Compute gate-weighted sum of all expert outputs. Returns (B, expert_dim)."""
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, expert_dim)
        weights = gate(x).unsqueeze(-1)                                   # (B, n_experts, 1)
        return (expert_outs * weights).sum(dim=1)                         # (B, expert_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mixed_a = self._mix_experts(x, self.gate_a)
        mixed_b = self._mix_experts(x, self.gate_b)
        mixed_c = self._mix_experts(x, self.gate_c)

        return self.head_a(mixed_a), self.head_b(mixed_b), self.head_c(mixed_c)

    def forward_with_gates(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Training path: same gated outputs as forward() plus the softmax gate
        weights for all three tasks.

        Used by the training loop to compute entropy regularization without a
        second full expert-computation pass. Experts are computed once and reused
        across all three gates, making this marginally faster than forward() too.

        Returns: (head_a, head_b, head_c, gate_a_weights, gate_b_weights, gate_c_weights)
        Gate weight tensors are shape (B, n_experts) — already softmaxed.
        """
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        gate_a_w = self.gate_a(x)
        gate_b_w = self.gate_b(x)
        gate_c_w = self.gate_c(x)
        mixed_a = (expert_outs * gate_a_w.unsqueeze(-1)).sum(dim=1)
        mixed_b = (expert_outs * gate_b_w.unsqueeze(-1)).sum(dim=1)
        mixed_c = (expert_outs * gate_c_w.unsqueeze(-1)).sum(dim=1)
        return (
            self.head_a(mixed_a), self.head_b(mixed_b), self.head_c(mixed_c),
            gate_a_w, gate_b_w, gate_c_w,
        )

    def predict_with_diagnostics(self, x: Tensor) -> dict:
        """
        Inference-time path that returns the same gated outputs as forward()
        plus interpretability diagnostics:

          - "gates":    softmax weights per head — which experts each head
                        trusted for this input. Tuple of 3 tensors, each (B, n_experts).
          - "opinions": what each head WOULD predict if it trusted only one
                        expert. Tuple of 3 tensors:
                            head_a_opinions: (B, n_experts, 1)
                            head_b_opinions: (B, n_experts, n_traj)
                            head_c_opinions: (B, n_experts, n_hazard)

        Mathematically equivalent to forward() for the gated outputs (verified
        in tests). Faster than forward(), because experts are computed once
        instead of three times (once per head).

        Training is unaffected — forward() is unchanged. This method is only
        called from MMoEPredictor.predict() in the inference path.
        """
        # Compute every expert once and reuse for both gating and per-expert
        # opinions. (B, n_experts, expert_dim)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        # Gating softmax weights (B, n_experts) per head
        gate_a_w = self.gate_a(x)
        gate_b_w = self.gate_b(x)
        gate_c_w = self.gate_c(x)

        # Gated outputs — same math as forward(), just sharing expert_outs.
        mixed_a = (expert_outs * gate_a_w.unsqueeze(-1)).sum(dim=1)
        mixed_b = (expert_outs * gate_b_w.unsqueeze(-1)).sum(dim=1)
        mixed_c = (expert_outs * gate_c_w.unsqueeze(-1)).sum(dim=1)

        head_a_gated = self.head_a(mixed_a)
        head_b_gated = self.head_b(mixed_b)
        head_c_gated = self.head_c(mixed_c)

        # Per-expert opinions: feed each expert's hidden output through each
        # head individually. Heads have no BatchNorm and Dropout is a no-op
        # in eval mode, so each head is a deterministic function of its input
        # — these "opinions" are well-defined predictions, not approximations.
        n_experts = expert_outs.size(1)
        head_a_opinions = torch.stack(
            [self.head_a(expert_outs[:, i, :]) for i in range(n_experts)], dim=1
        )
        head_b_opinions = torch.stack(
            [self.head_b(expert_outs[:, i, :]) for i in range(n_experts)], dim=1
        )
        head_c_opinions = torch.stack(
            [self.head_c(expert_outs[:, i, :]) for i in range(n_experts)], dim=1
        )

        return {
            "gated":    (head_a_gated, head_b_gated, head_c_gated),
            "gates":    (gate_a_w, gate_b_w, gate_c_w),
            "opinions": (head_a_opinions, head_b_opinions, head_c_opinions),
        }
