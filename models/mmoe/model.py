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
