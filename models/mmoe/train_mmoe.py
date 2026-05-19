"""
MMoE Training Entry Point.

Usage:
    python -m models.mmoe.train_mmoe
    python -m models.mmoe.train_mmoe --smoke-test    # 5-game smoke test
    python -m models.mmoe.train_mmoe --lr 5e-4 --w-b 0.3
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.mmoe.dataset import (
    FEED_DELAY_SECONDS_NBA,
    FEED_DELAY_SECONDS_SPORTRADAR,
    build_dataloaders,
)
from models.mmoe.model import MMoEModel
from models.mmoe.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MMoE model")
    parser.add_argument("--smoke-test",   action="store_true", help="Run a quick sanity check (5 epochs, small data)")
    parser.add_argument("--max-epochs",   type=int,   default=200)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--w-a",           type=float, default=1.0,  help="Head A loss weight")
    parser.add_argument("--w-b",           type=float, default=1.0,  help="Head B loss weight (raised from 0.5 — Head B is the live entry signal)")
    parser.add_argument("--w-c",           type=float, default=0.3,  help="Head C loss weight")
    parser.add_argument("--lambda-entropy", type=float, default=0.02, help="Gate entropy regularization strength (0 = disabled)")
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--tp",           type=float, default=5.0,  help="Take-profit threshold (cents)")
    parser.add_argument("--sl",           type=float, default=3.0,  help="Stop-loss threshold (cents)")
    parser.add_argument("--feed-delay",   type=int,   default=FEED_DELAY_SECONDS_NBA,
                        help=f"Feed delay in seconds. NBA polling={FEED_DELAY_SECONDS_NBA}, "
                             f"Sportradar WS={FEED_DELAY_SECONDS_SPORTRADAR}")
    args = parser.parse_args()

    # Derive save paths from feed delay so runs don't overwrite each other
    suffix      = f"delay{args.feed_delay}"
    model_path  = Path(f"models/saved/mmoe_{suffix}.pt")
    scaler_path = Path(f"models/saved/mmoe_scaler_{suffix}.pkl")

    if args.smoke_test:
        logger.info("=== SMOKE TEST MODE ===")
        args.max_epochs = 5
        args.patience   = 999

    logger.info(
        "Feed delay: %ds | Saving model → %s | scaler → %s",
        args.feed_delay, model_path, scaler_path,
    )
    logger.info("Building dataloaders (this may take a few minutes)...")
    train_loader, val_loader, scaler = build_dataloaders(
        batch_size=args.batch_size,
        tp=args.tp,
        sl=args.sl,
        feed_delay_seconds=args.feed_delay,
    )

    logger.info(
        "Train batches: %d | Val batches: %d | Feature dim: %d",
        len(train_loader), len(val_loader),
        next(iter(train_loader))[0].shape[1],
    )

    model = MMoEModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model created: %d trainable parameters", n_params)

    result = train(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        max_epochs     = args.max_epochs,
        lr             = args.lr,
        w_a            = args.w_a,
        w_b            = args.w_b,
        w_c            = args.w_c,
        lambda_entropy = args.lambda_entropy,
        patience       = args.patience,
        save_path      = model_path,
    )

    # Save scaler alongside the model
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", scaler_path)

    # Final summary
    final_val = result.val_history[result.best_epoch - 1]
    logger.info("=== RESULTS [feed_delay=%ds | %s] ===", args.feed_delay, model_path.name)
    logger.info("  Best epoch: %d | Best val loss: %.4f", result.best_epoch, result.best_val_loss)
    logger.info("  Head A  AUCPR:      %.4f  (baseline 0.084, target >0.097)", final_val.aucpr_a)
    logger.info("  Head B  RMSE:       %.4f  (prev model: 28.75 log-odds delta)", final_val.rmse_b)
    logger.info("  Head B  Dir Acc:    %.3f  (prev model: 66.5%%)", final_val.dir_acc_b)
    logger.info("  Head C  Brier:      %.4f", final_val.brier_c)

    if final_val.aucpr_a < 0.097:
        logger.warning(
            "Head A AUCPR %.4f is more than 10%% below baseline (0.1075). "
            "Consider reducing w_b and w_c to let Head A learn more freely.",
            final_val.aucpr_a,
        )


if __name__ == "__main__":
    main()
