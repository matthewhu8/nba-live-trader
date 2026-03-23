---
description: Summarize and analyze the most recent backtest results. Run after any backtest to understand what happened and where to improve.
allowed-tools: Read, Glob
---

# Backtest Results Summary

Read the most recent results file from `backtesting/results/`.
Produce a structured analysis.

## Step 1: Load Results
Find the most recently modified file in `backtesting/results/`.
Load and read its full contents.

## Step 2: Performance Summary
Report these numbers clearly:
- Strategy name and parameters
- Date range covered
- Total trades, win rate
- Gross PnL, total fees paid, net PnL
- Fee drag as % of gross PnL (flag if >40%)
- Sharpe ratio, max drawdown
- Average hold time (minutes)

## Step 3: Context Breakdown
Summarize performance by:
- Quarter (Q1/Q2/Q3/Q4) — which quarters were profitable?
- Score differential bucket (close / medium / blowout)
- Run length at entry — better to enter early or late in a run?
- Shot quality (sustainable vs unsustainable scoring)

Identify the single best-performing context and worst-performing context.
These are the most actionable findings.

## Step 4: Signal Quality
If calibration data is available:
- How well-calibrated is the run prediction? (predicted 70% → actually 70%?)
- What % of predicted edge was actually captured?
- Which signal types had highest false positive rate?

## Step 5: Fee Drag Analysis
If fee drag > 40% of gross PnL:
- Too many small trades that barely clear fees
- Suggest raising the edge threshold or minimum position size
- Identify which strategy/context is generating the most fee drag

## Step 6: Red Flags
Check for and call out:
- Sharpe > 3.0 on training data = likely lookahead bias, investigate before trusting
- Win rate > 65% = suspicious, verify no data leakage
- All PnL concentrated in one quarter or one team = overfit to specific patterns
- Max drawdown > 30% of total PnL = risk management issue

## Step 7: Top 3 Improvements
Based on the context breakdown and red flags, suggest the 3 highest-priority
improvements to try in the next iteration. Be specific:
- "Add is_garbage_time filter — Q4 blowout trades are destroying Sharpe"
- "Raise edge threshold from 5% to 7% — too much fee drag on small-edge trades"
- "MeanReversionStrategy underperforms in Q1 — consider disabling for first quarter"