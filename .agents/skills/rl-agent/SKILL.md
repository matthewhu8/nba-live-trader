---
name: rl-agent
description: Working on the RL agent, reward design, state/action space, training loop, or the sequential trading decision model
---

# RL Agent Skill

## What the RL Agent Is For
The run_predictor.py answers: "is a run coming?"
The RL agent answers: "should I enter a position RIGHT NOW, how big, and when do I exit?"
These are sequential decisions under uncertainty — the RL problem.

The agent learns things that hard-coded rules cannot capture:
- Enter early in the run signal (Kalshi reprices slowly — timing matters)
- Don't enter when score_diff is already large (blowout suppresses price movement)
- This lineup signal is reliable Q2 but not Q4
- Shot quality matters more than run length at entry
- When to cut a losing position vs. hold through noise

## Start Simple — Understand Before Complexifying
Build in this order. Do not skip ahead.
1. Contextual bandit (no sequential state — just: given these features, what action?)
2. Tabular Q-learning (add sequential state, small discrete state space)
3. DQN or PPO only if tabular approach is clearly insufficient
Most of the edge comes from the feature engineering, not the RL architecture.
A well-trained Q-table beats a poorly-specified neural network every time.

## State Space
```python
@dataclass
class AgentState:
    run_probability: float        # output of run_predictor.py (0.0-1.0)
    kalshi_price: int             # current YES price in cents (1-99)
    lineup_net_rating_delta: float
    current_run_length: int
    scoring_sustainable: bool
    score_diff: int
    quarter: int
    time_remaining_seconds: int
    current_position: int         # contracts held (-N to +N, 0 = flat)
    time_in_position_seconds: int
    unrealized_pnl_cents: float
```
Discretize continuous variables for tabular approach.
Keep state space small until you have evidence it needs to be larger.

## Action Space
```python
class Action(Enum):
    BUY_YES_SMALL  = 0   # e.g. 10 contracts
    BUY_YES_LARGE  = 1   # e.g. 25 contracts
    BUY_NO_SMALL   = 2
    BUY_NO_LARGE   = 3
    EXIT           = 4   # close current position
    WAIT           = 5   # do nothing
```
Start with just: BUY_YES, BUY_NO, EXIT, WAIT (4 actions).
Add sizing actions only after basic policy is learned.

## Reward Signal
```python
def compute_reward(fill: OrderFill | None, position: Position | None) -> float:
    if fill is not None:
        # Reward on close: net PnL after maker fees
        return fill.realized_pnl - (0.0175 * fill.contracts * fill.price / 100)
    if position is not None and position.time_held > MAX_HOLD_SECONDS:
        # Penalty for holding too long — encourages timely exits
        return -0.5
    return 0.0  # no reward for intermediate steps
```
Sparse rewards are fine here — we only care about closed position PnL.
Do not reward on unrealized PnL — this creates holding bias.
Penalize excessive holding time to discourage getting stuck in bad positions.

## Training Loop
Train entirely through `backtesting/simulator.py`.
Never train on live data.
```python
for episode in range(N_EPISODES):
    game = sample_game(training_set)     # from 2021-22 or 2022-23 seasons only
    simulator.run_game(game, agent)      # agent makes decisions, gets rewards
    agent.update_policy()                # Q-table update or gradient step
```
Episodes = individual games. Shuffle game order during training (within season split).
Evaluate on validation set (2023-24) periodically to check for overfitting.

## Overfitting Warning Signs
- Agent performs great on training games, poor on validation = overfit
- Agent learns to never trade (reward 0 > negative reward) = reward shaping issue
- Agent always takes BUY_YES regardless of features = not learning state properly
- Sharpe collapses in validation relative to training = state space too large for data

## Integration with Run Predictor
```python
run_prob = run_predictor.predict(features)
agent_state = AgentState(
    run_probability=run_prob,
    kalshi_price=features.kalshi_yes_ask,
    ...
)
action = agent.select_action(agent_state)
```
The agent is downstream of the predictor. It uses run_prob as one input among many.
The agent can choose WAIT even when run_prob is high (e.g. garbage time, small edge).

## Do Not Build This Until
- Feature store is complete and validated (no lookahead bias)
- run_predictor.py is trained and shows positive signal on validation set
- backtesting/simulator.py is complete with at least 2 working strategies
- You understand what a "good" backtest looks like for this domain
Building the RL agent on top of a broken feature store wastes weeks.