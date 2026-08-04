# Live Trading Execution

## Paper Trading (Two-Process Setup)

**Terminal 1** — Python inference service (start first):
```bash
PYTHONPATH=.:live-trader ./venv/bin/python -m uvicorn inference.main:app \
  --host 127.0.0.1 --port 8001
```
Wait for "Application startup complete."

**Terminal 2** — Go trader:
```bash
cd live-trader/go
go build . && ./go                    # auto-detects today's games
./go --game 0042500223               # single-game mode
./go --event KXNBASPREAD-26MAY09OKCLAL
```

Dashboard: http://127.0.0.1:8001/dashboard

## Inspection & Logs
```bash
# Full post-game summary
./venv/bin/python tools/inspect_run.py <run_id>

# Live logs
tail -f live-trader/go/logs/runs/{date}/{run_id}/live-trader.jsonl
tail -f live-trader/go/logs/runs/{date}/{run_id}/inference.jsonl
```

## Configuration
- Entry thresholds in `live-trader/config/trading.yaml`
- Edit config, restart Go (no recompile needed)
- Paper mode is default. Confirm before switching to live.

## Gate inputs are NOT model features
Anything the Go agent gates on must be a **named field** on `PossessionResponse`
(`is_overtime`, `current_run_length`, `is_garbage_time`, `is_blowout`, …), never a lookup in
`resp.Features`. That map is **logging only**.

Go returns 0 for a missing map key with no error, so a feature rename silently disables a gate.
This already happened: removing `period` and `current_run_length` from the feature set turned
off the overtime skip and blocked every entry. Adding a gate input means editing
`inference/main.py`, `go/inference_client.go`, and `go/agent.go` together.

Guarded by `test_go_never_gates_on_a_feature_lookup` in `tests/test_feature_parity.py`.

## Credentials
- `KALSHI_KEY_ID` and `KALSHI_PEM_PATH` in `.env`
- PEM stored outside repo (`~/.kalshi/private_key.pem`)
- Never commit `.env`
