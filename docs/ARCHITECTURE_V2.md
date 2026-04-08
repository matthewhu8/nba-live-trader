# Kalshi Basketball Trading System - v2 Architecture (MMoE & Hybrid Execution)

This document completely replaces the Layer 1/Layer 2 XGBoost cascaded pipeline outlined in the original structure. It covers the full-stack mathematical, architectural, and data engineering logic required for the Multi-Task Learning (MTL) and Multi-gate Mixture-of-Experts (MMoE) framework.

---

## 1. The Core Problem with v1 Architecture
The previous system utilized two isolated models (the "telephone game" flaw). Layer 1 predicted a physics-based probability (P_run), and passed that static number to Layer 2, which predicted Kalshi price momentum. 
*   **Cascading Errors**: Layer 2 treated Layer 1's output as absolute truth. If Layer 1 miscalculated, Layer 2 learned the error as noise.
*   **Static Horizons**: Layer 2 predicted a static 120s hold-time expected value, ignoring momentum stalling or catastrophic "hazard" drawdowns during the hold.
*   **No Position Sizing Intuition**: Trades were entered blindly without mapping probabilities to the Kelly Criterion or dynamically adjusting stops.

---

## 2. Data Engineering & The Master State Vector

The new architecture requires unifying two datasets that operate on different clock schemas. We synchronize them at the exact millisecond of the decision point.

### The Input Space (X)
For every single prediction decision, the model calculates the unified master vector `X = [X_physics, X_market]`.
*   **X_physics (Basketball Reality)**: Sourced from `possession_flat`. Contains matchup deltas, coach rotation flags, star foul trouble, running shot quality, and clock momentum.
*   **X_market (Finance Reality)**: Sourced from `kalshi_ticks`. Contains bid/ask spreads, current order book depth and queue imbalances, and last 60s traded volume.

### The Target Labels
When evaluating the historical data to generate the "True Answers", we no longer label rows with arbitrary static truths. We simulate the logic.
Target arrays are mapped:
*   `True_Run`: Did a 6+ point scoring play occur within the next 5 possessions? (1 or 0)
*   `True_Trajectory`: Array of observed prices at future offsets relative to entry `[P_30s, P_60s, P_90s, P_120s]`.
*   `True_Momentum_Flip`: Array denoting exactly which possession the run completely evaporated on historically.

---

## 3. The Core ML Architecture: Multi-gate Mixture-of-Experts (MMoE)

Instead of passing predictions between models, the raw master vector `X` is fed into a single, unified deep learning architecture (typically PyTorch).

### A. The Shared Features (The Experts)
We construct `K` parallel feedforward networks (e.g., 3 to 5 "Experts"). Each Expert functions independently across the full dimension of `X`. 
Over time, Backpropagation mathematically forces each expert to organically specialize (e.g., Expert 1 masters lineups, Expert 2 masters Kalshi bid walls).

### B. The Managers (The Gating Networks)
We construct three independent Task Managers. For each input row, the Manager assigns a Softmax percentage budget across the K Experts.
*   **Gate A (Physics)**: Focuses purely on Experts looking at physical basketball mismatch data.
*   **Gate B (Price Trajectories)**: Blends physics Experts with LOB structure Experts.
*   **Gate C (Stall Hazards)**: Heavily weights momentum sequence Experts to predict game stalls.

### C. The Three Output Heads
1.  **Head A (Physics Event Classifier)**: Outputs the true underlying probability that a multi-possession scoring run will physically occur on the court in the next 3 real minutes. (Binary Classification)
2.  **Head B (Multi-Horizon Price Regressor)**: Outputs the expected Kalshi future price at specific time gates: `[E_30s, E_60s, E_90s, E_120s]`. (Multi-Output Regression)
3.  **Head C (Survival / Hazard predictor)**: Outputs the array of risks that the sequence breaks over the next N possessions `[Haz_1, Haz_2, Haz_3, Haz_4]`. (Discrete-Time Survival Analysis)

---

## 4. The Joint Loss Function (The Mathematical Regularizer)

We train the entire network end-to-end to minimize a unified loss function.

`Total_Loss = (Weight_A * CrossEntropy_Loss_A) + (Weight_B * HuberLoss_Trajectory_B) + (Weight_C * LogLikelihood_Hazard_C)`

### Why this structure prevents blowing up:
If the Kalshi market exhibits high-variance noise (i.e. glitchy tick prints or spoofing), the hidden Experts will immediately try to memorize those patterns. 
However, **Head A (Physics Event)** grounds those Experts. If the Experts alter their math to memorize market noise, their accuracy on predicting physical basketball tanks, and the CrossEntropy penalty triggers heavily during Backpropagation. 
Therefore, the Experts are mathematically forced to only identify Kalshi market patterns that are directly grounded in physical basketball phenomena. 

---

## 5. Execution Layer & Downside Protection

Once the Neural Network evaluates the master array `X` and passes the resulting arrays to the Execution Agent, the Agent triggers rigid EV-calculated logic.

### Dynamic Stop Loss and Take Profit
*   **Dynamic Take Profit (TP)** = Evaluates the Head B Trajectory. If the peak predicted price occurs at 60 seconds (e.g. 51 cents) before degrading, TP is automatically set near the peak (e.g. 50.5 cents) to secure profit prior to the degradation.
*   **Dynamic Stop Loss (SL)** = Tied to Head C. If Survival Probability drops below an acceptable baseline, the agent conditionally sets the Stop Loss to scratch the trade if the very next possession fails.

### Expected Value (EV) Calculation
`EV = (Prob_Hit_TP * Profit_If_Hit) - (Prob_Hit_SL * Loss_If_Hit) - Maker_Fees`
If EV <= 0, the trade is rejected entirely.

### Dynamic Position Sizing (Fractional Kelly)
Position size scales linearly with how much mathematical edge the model identifies, bounded heavily by market thinness.

1.  **Calculate Base Kelly Fraction**: `EV / Profit_If_Hit`
2.  **Apply Buffers and Limits**:
    *   Multiply base quantity by 0.5 (Half-Kelly Constraint against model noise).
    *   Multiply by `(1 - Expected_Hazard_Rate)` to shrink positions during chaotic, high-turnover game environments.
    *   **The LOB Constraint Rule:** Your final trade size can mathematically **never** exceed 20% of the currently available "maker maker" liquidity limits on Kalshi. 

### Hybrid Exit Execution (Event Loop)
The agent executes a market limit exit when the **earliest** of any condition is met:
1. Limit order reaches `Dynamic_TP`.
2. Limit order reaches `Dynamic_SL`.
3. The opposing team scores twice (canceling momentum physics).
4. The simulation time-gate expires (120 seconds game clock limit hit).
5. The game transitions to Blowout/Garbage time resulting in uncorrelated data mechanics.
