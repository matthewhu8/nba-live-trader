"""
Kalshi order book feature builder.

Processes high-frequency Kalshi tick data into microstructure features.
Input is expected to be a DataFrame of raw tick updates.
Outputs a DataFrame with derived features like spread, imbalance, and volume.
"""

import numpy as np
import pandas as pd


def add_kalshi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Kalshi tick data into order book microstructure features.

    Expected input columns:
      - ts (datetime): timestamp of the tick
      - bid (float): best bid price (cents)
      - ask (float): best ask price (cents)
      - bid_size (int): contracts available at best bid
      - ask_size (int): contracts available at best ask
      - volume (int): traded volume on this tick (if any, default 0)

    Generated features:
      - spread
      - book_imbalance
      - trade_volume_60s
      - time_since_last_trade_ms
    """
    out = df.copy()

    # Ensure ts is sorted and is datetime
    if "ts" in out.columns:
        out = out.sort_values("ts").reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(out["ts"]):
            out["ts"] = pd.to_datetime(out["ts"])

    # 1. Spread Compression
    if "yes_ask" in out.columns and "yes_bid" in out.columns:
        out["spread"] = out["yes_ask"] - out["yes_bid"]
        out["spread"] = out["spread"].clip(lower=1.0) # Kalshi minimum tick size

    if "ts" in out.columns:
        # 3. Liquidity Regime (Trade Volume over last 60s)
        if "volume" in out.columns:
            # Set index to timestamp for rolling window
            temp = out.set_index("ts")
            # 60-second rolling sum of volume
            volume_60s = temp["volume"].rolling("60s").sum().fillna(0)
            out["trade_volume_60s"] = volume_60s.values
        else:
            out["trade_volume_60s"] = 0.0

        # 4. Time Since Last Trade (Liquidity Proxy)
        if "volume" in out.columns:
            # Mark timestamps where a trade occurred
            trade_mask = out["volume"] > 0
            # Forward fill the last trade timestamp
            last_trade_ts = out["ts"].where(trade_mask).ffill()
            # If no trade has happened yet, assume long ago or start of data
            last_trade_ts = last_trade_ts.fillna(out["ts"].min())
            
            delta = out["ts"] - last_trade_ts
            out["time_since_last_trade_ms"] = delta.dt.total_seconds() * 1000
        else:
            out["time_since_last_trade_ms"] = 0.0

    return out
