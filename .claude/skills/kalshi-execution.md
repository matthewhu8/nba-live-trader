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

## Credentials
- `KALSHI_KEY_ID` and `KALSHI_PEM_PATH` in `.env`
- PEM stored outside repo (`~/.kalshi/private_key.pem`)
- Never commit `.env`
