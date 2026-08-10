"""
MMoE Training Loop.

Trains the MMoE model with a masked joint loss:
  loss = w_A * BCE(head_a, target_run, mask=has_run_target)
       + w_B * Huber(head_b, target_trajectory, mask=has_market_data)
       + w_C * BCE(head_c, target_hazard, mask=has_run_target)

Per-row masks ensure:
  - Basketball-only rows contribute to Heads A + C only (no market target noise)
  - Joint rows contribute to all three heads

Evaluation metrics per head:
  - Head A: AUCPR  (comparison baseline: run_predictor AUCPR 0.1075)
  - Head B: RMSE on log-odds delta, directional accuracy on tradable rows
  - Head C: Brier score per horizon (mean across horizons)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from models.mmoe.model import MMoEModel

logger = logging.getLogger(__name__)

SAVED_MODEL_DIR = Path("models/saved")

# DataLoader tensor indices (must match _build_tensor_dataset order)
IDX_X       = 0
IDX_RUN     = 1
IDX_TRAJ    = 2
IDX_HAZ     = 3
IDX_MKT     = 4   # has_market_data (float)
IDX_HAS_RUN = 5   # has_run_target  (float)


@dataclass
class EpochMetrics:
    loss_total: float = 0.0
    loss_a:     float = 0.0
    loss_b:     float = 0.0
    loss_c:     float = 0.0
    aucpr_a:    float = 0.0
    rmse_b:     float = 0.0
    dir_acc_b:  float = 0.0
    brier_c:    float = 0.0

    # Head B comparability instrumentation. `loss_b` is a Huber average taken over rows
    # selected by a threshold on the labels themselves (`_compute_loss`: mean |traj| >
    # 0.02), so a change to the exit simulator moves both the target values and the row
    # population. A raw loss_b or rmse_b from before such a change is therefore not
    # comparable to one from after — neither the quantity nor the denominator held still.
    # These three make the comparison recoverable after the fact:
    head_b_rows:           int = 0    # rows clearing the mask (the hidden variable)
    head_b_rows_total:     int = 0    # market rows considered, for the retention rate
    rmse_b_zero:           float = 0.0    # predict-nothing reference, gives rmse_b a scale
    rmse_b_per_checkpoint: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    best_val_loss:  float
    best_epoch:     int
    train_history:  list[EpochMetrics] = field(default_factory=list)
    val_history:    list[EpochMetrics] = field(default_factory=list)


def _masked_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE loss applied only to rows where mask=1. Returns scalar."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.functional.binary_cross_entropy(pred[mask > 0], target[mask > 0])


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Huber loss applied only to rows where mask=1. Returns scalar."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.functional.huber_loss(pred[mask > 0], target[mask > 0])


def _gate_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy of gate softmax weights across the batch.

    weights: (B, n_experts) — already softmaxed, values in (0, 1].
    Returns a scalar >= 0. Higher = more diverse routing across experts.

    Used to build the entropy regularization term: subtracting lambda * entropy
    from the total loss makes the optimizer prefer high-entropy (diverse) gates.
    """
    return -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()


def _compute_loss(
    head_a: torch.Tensor,
    head_b: torch.Tensor,
    head_c: torch.Tensor,
    batch: tuple,
    w_a: float,
    w_b: float,
    w_c: float,
    gate_a: Optional[torch.Tensor] = None,
    gate_b: Optional[torch.Tensor] = None,
    gate_c: Optional[torch.Tensor] = None,
    lambda_entropy: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute joint masked loss. Returns (total, loss_a, loss_b, loss_c).

    Head B mask: only rows with market data AND a meaningful trajectory signal.
    Roughly 42% of joint rows have traj_9 == 0 (time_gate/garbage_time exits at
    the entry price) — including these in the Huber loss biases Head B toward zero.
    Threshold 0.02 ≈ 0.5¢ logit-delta; TP/SL exits are ~0.12–0.20, so this only
    removes near-zero noise rows.
    """
    target_run  = batch[IDX_RUN].unsqueeze(1)  # (B, 1)
    target_traj = batch[IDX_TRAJ]              # (B, 10)
    target_haz  = batch[IDX_HAZ]               # (B, 10)
    has_mkt     = batch[IDX_MKT]               # (B,)
    has_run     = batch[IDX_HAS_RUN]           # (B,)

    # Exclude near-zero-trajectory rows from Head B: keep only rows where the mean
    # absolute logit-delta across all 10 checkpoints exceeds 0.02.
    traj_signal = (target_traj.abs().mean(dim=1) > 0.02).float()  # (B,)
    head_b_mask = (has_mkt * traj_signal).unsqueeze(1).expand_as(head_b)

    loss_a = _masked_bce(head_a.squeeze(1), target_run.squeeze(1), has_run)
    loss_b = _masked_huber(head_b, target_traj, head_b_mask)
    loss_c = _masked_bce(head_c.view(-1), target_haz.view(-1), has_run.unsqueeze(1).expand_as(head_c).reshape(-1))

    total = w_a * loss_a + w_b * loss_b + w_c * loss_c

    # Entropy regularization: subtract lambda * mean gate entropy from total loss.
    # Subtracting entropy (a positive number) means higher entropy → lower loss →
    # the optimizer actively seeks diverse gate routing, preventing collapse where
    # one expert dominates and the others stop receiving gradients.
    if lambda_entropy > 0.0 and gate_a is not None and gate_b is not None and gate_c is not None:
        H = _gate_entropy(gate_a) + _gate_entropy(gate_b) + _gate_entropy(gate_c)
        total = total - lambda_entropy * H

    return total, loss_a, loss_b, loss_c


@torch.no_grad()
def _evaluate(
    model: MMoEModel,
    loader: DataLoader,
    device: torch.device,
    w_a: float,
    w_b: float,
    w_c: float,
) -> EpochMetrics:
    model.eval()
    total_loss = loss_a_sum = loss_b_sum = loss_c_sum = 0.0
    n_batches = 0

    all_run_pred:  list[np.ndarray] = []
    all_run_true:  list[np.ndarray] = []
    all_traj_pred: list[np.ndarray] = []
    all_traj_true: list[np.ndarray] = []
    all_haz_pred:  list[np.ndarray] = []
    all_haz_true:  list[np.ndarray] = []
    all_mkt_mask:  list[np.ndarray] = []
    all_run_mask:  list[np.ndarray] = []

    for batch in loader:
        batch = [t.to(device) for t in batch]
        X = batch[IDX_X]
        head_a, head_b, head_c = model(X)

        total, la, lb, lc = _compute_loss(head_a, head_b, head_c, batch, w_a, w_b, w_c)
        total_loss  += total.item()
        loss_a_sum  += la.item()
        loss_b_sum  += lb.item()
        loss_c_sum  += lc.item()
        n_batches   += 1

        all_run_pred.append(head_a.squeeze(1).cpu().numpy())
        all_run_true.append(batch[IDX_RUN].cpu().numpy())
        all_traj_pred.append(head_b.cpu().numpy())
        all_traj_true.append(batch[IDX_TRAJ].cpu().numpy())
        all_haz_pred.append(head_c.cpu().numpy())
        all_haz_true.append(batch[IDX_HAZ].cpu().numpy())
        all_mkt_mask.append(batch[IDX_MKT].cpu().numpy())
        all_run_mask.append(batch[IDX_HAS_RUN].cpu().numpy())

    run_pred  = np.concatenate(all_run_pred)
    run_true  = np.concatenate(all_run_true)
    run_mask  = np.concatenate(all_run_mask).astype(bool)
    traj_pred = np.concatenate(all_traj_pred)
    traj_true = np.concatenate(all_traj_true)
    mkt_mask  = np.concatenate(all_mkt_mask).astype(bool)
    haz_pred  = np.concatenate(all_haz_pred)
    haz_true  = np.concatenate(all_haz_true)

    # Head A: AUCPR
    aucpr = 0.0
    if run_mask.sum() > 0 and run_true[run_mask].sum() > 0:
        try:
            aucpr = average_precision_score(run_true[run_mask], run_pred[run_mask])
        except ValueError:
            pass

    # Head B: RMSE + directional accuracy (on market rows only)
    # For directional accuracy, exclude rows where traj_true[:, -1] is near zero
    # (logit-delta ~0 means no price move — direction is undefined/noise for those).
    rmse_b = dir_acc_b = 0.0
    if mkt_mask.sum() > 0:
        diff   = traj_pred[mkt_mask] - traj_true[mkt_mask]
        rmse_b = float(np.sqrt((diff ** 2).mean()))
        # Only evaluate direction on rows with a meaningful final-checkpoint signal.
        # Threshold 0.05 ≈ ~1¢ logit-delta at 50¢; TP=+5¢ → ~0.20, SL=-3¢ → ~-0.12.
        dir_rows = np.abs(traj_true[mkt_mask, -1]) > 0.05
        if dir_rows.sum() > 0:
            pred_dir = traj_pred[mkt_mask][dir_rows, -1]
            true_dir = traj_true[mkt_mask][dir_rows, -1]
            correct_dir = np.sign(pred_dir) == np.sign(true_dir)
            dir_acc_b   = float(correct_dir.mean())

    # Head C: mean Brier score across all horizons (on run-target rows)
    brier_c = 0.0
    if run_mask.sum() > 0:
        brier_c = float(((haz_pred[run_mask] - haz_true[run_mask]) ** 2).mean())

    # Head B comparability instrumentation — see EpochMetrics. Computed against the same
    # mask `_compute_loss` optimises (has_market_data AND mean |traj| > 0.02), NOT the
    # looser `mkt_mask` used for rmse_b above, because the traj_signal term is the part
    # that shifts when the labels change.
    #
    # Per-checkpoint RMSE matters specifically here: at a 20s feed delay the old labels
    # put traj_0 and traj_1 at 12s and 24s, i.e. straddling the entry anchor, so those two
    # carried the most contamination. Aggregate RMSE averages away exactly the effect
    # under test.
    traj_signal       = np.abs(traj_true).mean(axis=1) > 0.02
    head_b_mask       = mkt_mask & traj_signal
    head_b_rows       = int(head_b_mask.sum())
    head_b_rows_total = int(mkt_mask.sum())
    rmse_b_zero = 0.0
    rmse_b_per_checkpoint: list[float] = []
    if head_b_rows > 0:
        masked_true = traj_true[head_b_mask]
        d = traj_pred[head_b_mask] - masked_true
        # RMSE of a model that predicts 0.0 everywhere: sqrt(mean(target^2)). If rmse_b
        # is not comfortably below this, Head B has learned nothing worth keeping.
        rmse_b_zero = float(np.sqrt((masked_true ** 2).mean()))
        rmse_b_per_checkpoint = [
            float(np.sqrt((d[:, k] ** 2).mean())) for k in range(d.shape[1])
        ]

    n = max(n_batches, 1)
    return EpochMetrics(
        loss_total = total_loss / n,
        loss_a     = loss_a_sum / n,
        loss_b     = loss_b_sum / n,
        loss_c     = loss_c_sum / n,
        aucpr_a    = aucpr,
        rmse_b     = rmse_b,
        dir_acc_b  = dir_acc_b,
        brier_c    = brier_c,
        head_b_rows           = head_b_rows,
        head_b_rows_total     = head_b_rows_total,
        rmse_b_zero           = rmse_b_zero,
        rmse_b_per_checkpoint = rmse_b_per_checkpoint,
    )


def train(
    model: MMoEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    max_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    w_a: float = 1.0,
    w_b: float = 1.0,
    w_c: float = 0.3,
    lambda_entropy: float = 0.02,
    patience: int = 15,
    grad_clip: float = 1.0,
    save_path: Optional[Path] = None,
) -> TrainResult:
    """
    Train the MMoE model with masked joint loss and early stopping.

    Args:
        model:        MMoEModel instance
        train_loader: DataLoader from build_dataloaders()
        val_loader:   DataLoader from build_dataloaders()
        device:       torch device (auto-detected if None)
        max_epochs:   maximum training epochs
        lr:           AdamW learning rate
        weight_decay: L2 regularization
        w_a/b/c:      head loss weights (start: 1.0 / 0.5 / 0.3)
        patience:     early stopping patience (epochs without val improvement)
        grad_clip:    gradient clipping max_norm
        save_path:    where to save best model checkpoint (.pt file)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    save_path = save_path or (SAVED_MODEL_DIR / "mmoe.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_improve = 0
    result = TrainResult(best_val_loss=best_val_loss, best_epoch=0)

    logger.info(
        "Starting MMoE training on %s | max_epochs=%d patience=%d | "
        "w_a=%.1f w_b=%.1f w_c=%.1f lambda_entropy=%.3f",
        device, max_epochs, patience, w_a, w_b, w_c, lambda_entropy,
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = epoch_la = epoch_lb = epoch_lc = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = [t.to(device) for t in batch]
            X = batch[IDX_X]

            optimizer.zero_grad()
            head_a, head_b, head_c, gate_a, gate_b, gate_c = model.forward_with_gates(X)
            total, la, lb, lc = _compute_loss(
                head_a, head_b, head_c, batch, w_a, w_b, w_c,
                gate_a, gate_b, gate_c, lambda_entropy,
            )

            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += total.item()
            epoch_la   += la.item()
            epoch_lb   += lb.item()
            epoch_lc   += lc.item()
            n_batches  += 1

        n = max(n_batches, 1)
        train_metrics = EpochMetrics(
            loss_total = epoch_loss / n,
            loss_a     = epoch_la / n,
            loss_b     = epoch_lb / n,
            loss_c     = epoch_lc / n,
        )
        val_metrics = _evaluate(model, val_loader, device, w_a, w_b, w_c)
        scheduler.step(val_metrics.loss_total)

        result.train_history.append(train_metrics)
        result.val_history.append(val_metrics)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d [%.1fs] | train_loss=%.4f (A=%.4f B=%.4f C=%.4f) "
            "| val_loss=%.4f AUCPR=%.4f RMSE_B=%.4f DirAcc=%.3f Brier_C=%.4f",
            epoch, max_epochs, elapsed,
            train_metrics.loss_total, train_metrics.loss_a, train_metrics.loss_b, train_metrics.loss_c,
            val_metrics.loss_total, val_metrics.aucpr_a,
            val_metrics.rmse_b, val_metrics.dir_acc_b, val_metrics.brier_c,
        )

        # Head B mask retention and the predict-nothing reference. Logged every epoch
        # because RMSE_B above cannot be read on its own: the mask is a function of the
        # labels, so a "better" RMSE_B may only mean Head B is being scored on fewer,
        # easier rows. `vs_zero` under 1.0 means the head beats predicting nothing.
        if val_metrics.head_b_rows_total > 0:
            logger.info(
                "         Head B rows %d/%d (%.1f%% retained) | RMSE_B/zero-pred = %.3f",
                val_metrics.head_b_rows, val_metrics.head_b_rows_total,
                100.0 * val_metrics.head_b_rows / val_metrics.head_b_rows_total,
                (val_metrics.rmse_b / val_metrics.rmse_b_zero)
                if val_metrics.rmse_b_zero > 0 else float("nan"),
            )

        if val_metrics.loss_total < best_val_loss:
            best_val_loss = val_metrics.loss_total
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": best_val_loss}, save_path)
            logger.info("  ✓ New best val loss %.4f — saved to %s", best_val_loss, save_path)
            # Per-checkpoint RMSE beside the checkpoint it describes. The early entries
            # (traj_0/traj_1 at 12s/24s) sit closest to the entry anchor and are where a
            # feed-delay regression in the labels would show up first.
            if val_metrics.rmse_b_per_checkpoint:
                logger.info(
                    "    RMSE_B per checkpoint: %s",
                    " ".join(f"{v:.3f}" for v in val_metrics.rmse_b_per_checkpoint),
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    result.best_val_loss = best_val_loss
    result.best_epoch    = best_epoch
    logger.info("Training complete. Best val loss %.4f at epoch %d.", best_val_loss, best_epoch)
    return result
