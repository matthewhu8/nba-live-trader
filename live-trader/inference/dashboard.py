"""
Live dashboard + SSE stream + CSV export for the paper trading system.

Endpoints:
    GET  /dashboard                → HTML dashboard page
    GET  /game/{game_id}/stream    → Server-Sent Events stream of predictions
    GET  /game/{game_id}/export    → CSV download of all possession predictions
"""

import csv
import io
import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()

class TradeUpdate(BaseModel):
    action: str
    direction: str
    price: int
    size: int
    pnl: float = 0.0
    reason: str = ""
    # market_ticker is the Kalshi market subscribed at order time. Carried so
    # the dashboard can render the team actually backed ("BUY SAS") instead of
    # generic "BUY YES", and so a scanner swap arriving between order and
    # broadcast can't relabel an already-placed trade. Empty when Go didn't
    # send it (no event ticker / older Go binary).
    market_ticker: str = ""

# SSE subscribers: game_id → list of asyncio.Queue
_subscribers: dict[str, list[asyncio.Queue]] = {}


def broadcast_prediction(game_id: str, data: dict) -> None:
    """Push a prediction to all SSE subscribers for this game."""
    queues = _subscribers.get(game_id, [])
    dead = []
    for q in queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        queues.remove(q)


async def _event_generator(game_id: str, request: Request) -> AsyncGenerator[str, None]:
    """Yield SSE events for a game until the client disconnects."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.setdefault(game_id, []).append(queue)
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                data = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield f"data: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive {datetime.utcnow().isoformat()}\n\n"
    finally:
        _subscribers.get(game_id, []).remove(queue) if queue in _subscribers.get(game_id, []) else None


@router.get("/game/{game_id}/stream")
async def game_stream(game_id: str, request: Request):
    """SSE endpoint streaming live prediction data."""
    return StreamingResponse(
        _event_generator(game_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/game/{game_id}/export")
async def game_export(game_id: str):
    """Download all prediction records as CSV."""
    from inference.main import _games
    state = _games.get(game_id)
    if state is None:
        return {"error": f"Game {game_id} not found"}

    output = io.StringIO()
    writer = None

    for record in state.prediction_history:
        row = {
            "possession_id": record.possession_id,
            "wall_clock_ts": record.wall_clock_ts.isoformat(),
            "run_prob": record.run_prob,
            "yes_bid": record.yes_bid,
            "yes_ask": record.yes_ask,
            "action": record.action,
        }
        # Add trajectory columns
        for i, v in enumerate(record.trajectory):
            row[f"traj_{i}"] = v
        # Add hazard columns
        for i, v in enumerate(record.hazard):
            row[f"hazard_{i}"] = v
        # Add all features
        row.update(record.features)

        if writer is None:
            writer = csv.DictWriter(output, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)

    output.seek(0)
    filename = f"{game_id}_{datetime.utcnow().strftime('%Y%m%d')}_predictions.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/game/{game_id}/trade")
async def report_trade(game_id: str, trade: TradeUpdate):
    """Receive trade execution from Go engine and broadcast to dashboard."""
    broadcast_prediction(game_id, {
        "type": "trade",
        "trade": trade.model_dump()
    })
    return {"status": "ok"}


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the live dashboard HTML page."""
    return DASHBOARD_HTML


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Paper Trader — Live</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e17;--s1:#111827;--s2:#1a2235;--bd:#2a3548;--t:#e5e7eb;--td:#9ca3af;--g:#22c55e;--r:#ef4444;--y:#eab308;--p:#a855f7;--c:#06b6d4;--b:#3b82f6;--nyk:#f97316;--phi:#3b82f6}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--t);min-height:100vh;padding:16px}
.hdr{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;background:var(--s1);border:1px solid var(--bd);border-radius:12px;margin-bottom:14px}
.hdr h1{font-size:18px;font-weight:700;display:flex;align-items:center;gap:10px}
.matchup{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;letter-spacing:.05em;color:var(--td)}
.matchup .vs{color:var(--td);font-weight:400;margin:0 8px}
.matchup .away{color:var(--c)}
.matchup .home{color:var(--nyk)}
.dot{width:10px;height:10px;border-radius:50%;background:var(--r);animation:p 2s infinite}
.dot.on{background:var(--g)}
@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}
.badges{display:flex;gap:8px}
.bdg{font-family:'JetBrains Mono',monospace;font-size:12px;padding:4px 10px;border-radius:6px;background:var(--s2);border:1px solid var(--bd)}
.conn{display:flex;align-items:center;gap:8px;margin-bottom:14px}
.conn input{font-family:'JetBrains Mono',monospace;font-size:14px;padding:8px 14px;border-radius:8px;border:1px solid var(--bd);background:var(--s1);color:var(--t);width:200px}
.conn button{padding:8px 18px;border-radius:8px;border:none;background:var(--b);color:#fff;font-weight:600;cursor:pointer;font-size:14px}
.conn button:hover{background:#2563eb}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.g2{display:grid;grid-template-columns:2fr 1fr;gap:14px}
.cd{background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:16px}
.ct{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--td);margin-bottom:10px}

/* Signal banner */
.signal{text-align:center;padding:20px;border-radius:12px;background:var(--s2);margin-bottom:14px;border:2px solid var(--bd);transition:all .5s}
.signal.buy{background:rgba(34,197,94,.12);border-color:var(--g)}
.signal.exit{background:rgba(239,68,68,.12);border-color:var(--r)}
.signal-text{font-size:36px;font-weight:800;font-family:'JetBrains Mono',monospace}
.signal-sub{font-size:13px;color:var(--td);margin-top:4px}

/* Run prob */
.rp-val{font-family:'JetBrains Mono',monospace;font-size:44px;font-weight:700;text-align:center;line-height:1}
.rp-bar{width:100%;height:8px;background:var(--s2);border-radius:4px;margin-top:10px;position:relative;overflow:hidden}
.rp-fill{height:100%;border-radius:4px;transition:width .5s,background .5s}
.rp-thresh{position:absolute;top:-4px;width:2px;height:16px;background:var(--y)}
.rp-lbl{font-size:11px;color:var(--td);margin-top:4px;text-align:center}

/* Price direction */
.price-dir{text-align:center;padding:10px 0}
.price-arrow{font-size:48px;line-height:1;transition:all .3s}
.price-label{font-size:14px;font-weight:600;margin-top:4px}
.price-detail{font-size:11px;color:var(--td);margin-top:2px;font-family:'JetBrains Mono',monospace}

/* Momentum */
.mom-meter{display:flex;gap:4px;align-items:flex-end;height:80px;justify-content:center;margin-top:8px}
.mom-bar{width:20px;border-radius:3px 3px 0 0;transition:height .4s,background .4s}
.mom-label{text-align:center;font-size:13px;font-weight:600;margin-top:8px}

/* Market */
.mkt{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.mkt-item{text-align:center}
.mkt-item .v{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700}
.mkt-item .l{font-size:11px;color:var(--td);margin-top:2px}

/* Log */
.log{max-height:200px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.8}
.log::-webkit-scrollbar{width:4px}
.log::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
.ll{padding:2px 0;border-bottom:1px solid var(--bd);white-space:nowrap}

/* Totals */
.tots{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.tot{text-align:center;padding:10px;background:var(--s2);border-radius:8px}
.tot .v{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700}
.tot .l{font-size:11px;color:var(--td);margin-top:3px}

.chart-c{height:100px;position:relative}
canvas{width:100%!important;height:100%!important}
</style>
</head>
<body>
<div class="hdr">
  <h1><span class="dot" id="dot"></span> Live Paper Trader</h1>
  <div class="matchup" id="matchup">— @ —</div>
  <div class="badges">
    <span class="bdg" id="bPoss">POSS: 0</span>
    <span class="bdg" id="bPipe">--ms</span>
  </div>
</div>

<div class="conn">
  <input id="gid" placeholder="Game ID (e.g. 0042500212)"/>
  <button onclick="go()">Connect</button>
  <span id="glab" style="color:var(--td);font-size:13px;margin-left:8px"></span>
</div>

<!-- Signal Banner -->
<div class="signal" id="sig">
  <div class="signal-text" id="sigTxt">WAITING</div>
  <div class="signal-sub" id="sigSub">Connect to a game to begin</div>
</div>

<!-- Row 1: Run Prob | Price Direction | Momentum -->
<div class="g3">
  <div class="cd">
    <div class="ct">🏀 Scoring Run Probability</div>
    <div class="rp-val" id="rp" style="color:var(--td)">--</div>
    <div class="rp-bar"><div class="rp-fill" id="rpF" style="width:0"></div><div class="rp-thresh" style="left:10%"></div></div>
    <div class="rp-lbl">Buy threshold: 10% · <span id="runTeam" style="font-weight:600">No active run</span></div>
  </div>

  <div class="cd">
    <div class="ct">📈 Market Price Direction (next 2 min)</div>
    <div class="price-dir">
      <div class="price-arrow" id="pArr">—</div>
      <div class="price-label" id="pLbl" style="color:var(--td)">Waiting for data</div>
      <div class="price-detail" id="pDet"></div>
    </div>
  </div>

  <div class="cd">
    <div class="ct">💪 Momentum Strength</div>
    <div class="mom-meter" id="momM"></div>
    <div class="mom-label" id="momL" style="color:var(--td)">--</div>
  </div>
</div>

<!-- Row 2: Market -->
<div class="cd" style="margin-bottom:14px">
  <div class="ct" id="mktTitle">Market Orderbook</div>
  <div class="mkt">
    <div class="mkt-item"><div class="v" id="mBid">--</div><div class="l">YES BID</div></div>
    <div class="mkt-item"><div class="v" id="mAsk">--</div><div class="l">YES ASK</div></div>
    <div class="mkt-item"><div class="v" id="mSpr">--</div><div class="l">SPREAD</div></div>
    <div class="mkt-item"><div class="v" id="mSD">--</div><div class="l">SCORE DIFF</div></div>
    <div class="mkt-item"><div class="v" id="mRun">--</div><div class="l">CURRENT RUN</div></div>
  </div>
</div>

<!-- Row 3: Log + Totals -->
<div class="g2">
  <div class="cd">
    <div class="ct">Live Possession Log</div>
    <div class="log" id="log"><div class="ll" style="color:var(--td)">Waiting for connection...</div></div>
  </div>
  <div class="cd">
    <div class="ct">Paper Trading Totals</div>
    <div class="tots">
      <div class="tot"><div class="v" id="tSig">0</div><div class="l">Signals</div></div>
      <div class="tot"><div class="v" id="tTrd">0</div><div class="l">Trades</div></div>
      <div class="tot"><div class="v" id="tPnl" style="color:var(--g)">$0</div><div class="l">Net P&L</div></div>
      <div class="tot"><div class="v" id="tWR">--</div><div class="l">Win Rate</div></div>
    </div>
  </div>
</div>

<!-- Row 4: Trade History -->
<div class="cd" style="margin-top:14px">
  <div class="ct">Trade Execution History</div>
  <div class="log" id="tradeLog">
    <div class="ll" style="color:var(--td)">No trades executed yet...</div>
  </div>
</div>

<script>
let es, sigs=0, trds=0, pnl=0, wins=0;
let homeTeam = 'HOME', awayTeam = 'AWAY';
let yesTeam = '', noTeam = '';     // who BUY_YES / BUY_NO is actually backing
let currentMarketTicker = '';      // updated from each SSE message

// teamsFromMarketTicker parses a Kalshi NBA spread market ticker into team metadata.
//
//   KXNBASPREAD-26MAY10SASMIN-SAS5     → {away:"SAS", home:"MIN", yes:"SAS", no:"MIN"}
//   KXNBASPREAD-26MAY10NYKPHI-PHI3.5   → {away:"NYK", home:"PHI", yes:"PHI", no:"NYK"}
//
// The event segment is fixed-width: YY + MMM + DD + AWAY3 + HOME3.
// The trailing market suffix is the team the spread is FOR, which is also the
// YES side of the contract. Returns null on any unexpected format so callers
// can fall back to generic HOME/AWAY labels.
function teamsFromMarketTicker(ticker) {
  const m = ticker && ticker.match(/^KXNBASPREAD-\d{2}[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})-([A-Z]{3})[\d.]*$/);
  if (!m) return null;
  const away = m[1], home = m[2], yes = m[3];
  const no = (yes === home) ? away : home;
  return { away, home, yes, no };
}

// applyTicker updates the in-memory team state from a freshly-arrived market
// ticker AND refreshes the visible matchup header. Called from both possession
// messages (sets up the game labels) and trade messages (race-proof: the
// trade-line label always reflects the market actually subscribed at order
// time, even if the scanner swapped between order and broadcast).
function applyTicker(ticker) {
  if (!ticker || ticker === currentMarketTicker) return false;
  const t = teamsFromMarketTicker(ticker);
  if (!t) return false;
  currentMarketTicker = ticker;
  homeTeam = t.home; awayTeam = t.away;
  yesTeam  = t.yes;  noTeam   = t.no;
  // Matchup header (visible across the top)
  const m = document.getElementById('matchup');
  if (m) m.innerHTML = `<span class="away">${t.away}</span><span class="vs">@</span><span class="home">${t.home}</span>`;
  document.getElementById('mktTitle').textContent = 'Market Orderbook (' + ticker + ')';
  return true;
}

function go(){
  const id=document.getElementById('gid').value.trim();
  if(!id)return;

  // Reset team labels — they'll auto-populate from the first market_ticker
  // SSE message via teamsFromMarketTicker. No more hardcoded game-ID maps.
  homeTeam = 'HOME'; awayTeam = 'AWAY';
  yesTeam = ''; noTeam = '';
  currentMarketTicker = '';
  document.getElementById('mktTitle').textContent = 'Market Orderbook (connecting…)';
  const mh = document.getElementById('matchup');
  if (mh) mh.innerHTML = '— @ —';

  if(es)es.close();
  document.getElementById('glab').textContent='Connecting...';
  document.getElementById('dot').className='dot';
  document.getElementById('log').innerHTML='';
  es=new EventSource('/game/'+id+'/stream');
  es.onopen=()=>{document.getElementById('dot').className='dot on';document.getElementById('glab').textContent='Live — '+id};
  es.onmessage=(e)=>upd(JSON.parse(e.data));
  es.onerror=()=>{document.getElementById('dot').className='dot'};
}

function upd(d){
  // Auto-derive team labels on the first message that carries a market ticker,
  // and again whenever the scanner swaps to a different market.
  applyTicker(d.market_ticker);

  // Handle trade events
  if (d.type === 'trade') {
    const t = d.trade;
    if(trds===0) document.getElementById('tradeLog').innerHTML=''; // clear placeholder

    // Resolve the team from the TRADE's own ticker so a late-arriving scanner
    // swap can't relabel an already-placed trade. Falls back to current
    // game-state labels for older Go binaries that don't send market_ticker.
    let tradeYesTeam = yesTeam, tradeNoTeam = noTeam;
    if (t.market_ticker) {
      const tt = teamsFromMarketTicker(t.market_ticker);
      if (tt) { tradeYesTeam = tt.yes; tradeNoTeam = tt.no; }
    }

    const ll = document.createElement('div');
    ll.className = 'll';
    const ts = new Date().toLocaleTimeString();

    if (t.action === 'ENTRY') {
      trds++; document.getElementById('tTrd').textContent = trds;
      const dColor = t.direction === 'YES' ? '#22c55e' : '#ef4444';
      const backing = t.direction === 'YES' ? (tradeYesTeam || 'YES') : (tradeNoTeam || 'NO');
      // Team is the dominant info; (YES/NO) is secondary. Ticker appended in
      // muted text so you can audit which exact contract was bought.
      ll.innerHTML = `<span style="color:#6b7280">${ts}</span> <span style="color:${dColor};font-weight:700;font-size:13px">BUY ${backing}</span> <span style="color:#6b7280">(${t.direction})</span> · ${t.size} @ ${t.price}¢ <span style="color:#6b7280">· ${t.market_ticker || ''}</span>`;
    } else {
      pnl += t.pnl;
      if (t.pnl > 0) wins++;
      const pnlColor = t.pnl > 0 ? '#22c55e' : '#ef4444';
      const pnlSign = t.pnl > 0 ? '+' : '';
      const wasBacking = t.direction === 'YES' ? (tradeYesTeam || 'YES') : (tradeNoTeam || 'NO');
      ll.innerHTML = `<span style="color:#6b7280">${ts}</span> <span style="color:#eab308;font-weight:700;font-size:13px">EXIT ${wasBacking}</span> <span style="color:#6b7280">(${t.reason})</span> · @ ${t.price}¢ · P&L: <span style="color:${pnlColor}">${pnlSign}$${t.pnl.toFixed(2)}</span>`;

      document.getElementById('tPnl').textContent = (pnl>0?'+':'')+'$'+pnl.toFixed(2);
      document.getElementById('tPnl').style.color = pnl>0?'#22c55e':'#ef4444';
      document.getElementById('tWR').textContent = ((wins/trds)*100).toFixed(0)+'%';
    }
    
    const log = document.getElementById('tradeLog');
    log.appendChild(ll); log.scrollTop = log.scrollHeight;
    return;
  }

  const rp=d.run_prob||0, f=d.features||{}, tr=d.trajectory||[], hz=d.hazard||[];

  // Signal banner — name the actual team being backed, not just YES/NO.
  const sig=document.getElementById('sig'), st=document.getElementById('sigTxt'), ss=document.getElementById('sigSub');
  if(rp>=0.10 && tr[9]>0){
    const team = yesTeam || 'YES';
    sig.className='signal buy';
    st.textContent='★ BUY ' + team;
    st.style.color='#22c55e';
    ss.textContent='Run detected + price rising → backing ' + team + ' to cover';
  } else if(rp>=0.10 && tr[9]<0){
    const team = noTeam || 'NO';
    sig.className='signal buy';
    st.textContent='★ BUY ' + team;
    st.style.color='#ef4444';
    ss.textContent='Run detected + price falling → backing ' + team;
  } else {
    sig.className='signal'; st.textContent='WAIT'; st.style.color='#9ca3af';
    ss.textContent='No scoring run detected · monitoring possessions';
  }

  // Run probability
  const rpE=document.getElementById('rp');
  rpE.textContent=(rp*100).toFixed(1)+'%';
  rpE.style.color=rp>=.10?'#22c55e':rp>=.05?'#eab308':'#9ca3af';
  document.getElementById('rpF').style.width=(rp*100)+'%';
  document.getElementById('rpF').style.background=rp>=.10?'#22c55e':rp>=.05?'#eab308':'#3b82f6';

  // Who's running
  const rt=f.current_run_team_encoded||0, rl=f.current_run_length||0, rpts=f.current_run_points||0;
  const rtE=document.getElementById('runTeam');
  if(rl>0){
    const who=rt>0?homeTeam:awayTeam;
    const clr=rt>0?'#f97316':'#3b82f6';
    rtE.innerHTML='<span style="color:'+clr+'">'+who+' on a '+rpts+'-pt run ('+rl+' poss)</span>';
  } else { rtE.textContent='No active run'; }

  // Price direction (Head B) — uses YES team, not home team. The home team
  // isn't necessarily the YES side (e.g., SAS @ MIN where the active market
  // is SAS+5, YES=SAS=away team). Falls back to "YES side" if we haven't
  // parsed a ticker yet.
  const pA=document.getElementById('pArr'), pL=document.getElementById('pLbl'), pD=document.getElementById('pDet');
  if(tr.length>0){
    const last=tr[9]||0, mid=tr[4]||0;
    const yLabel = yesTeam || 'YES side';
    if(Math.abs(last)<0.01){
      pA.textContent='→'; pA.style.color='#9ca3af';
      pL.textContent='Flat — no expected move'; pL.style.color='#9ca3af';
    } else if(last>0){
      pA.textContent='↑'; pA.style.color='#22c55e';
      pL.textContent=`${yLabel} price expected to RISE`; pL.style.color='#22c55e';
    } else {
      pA.textContent='↓'; pA.style.color='#ef4444';
      pL.textContent=`${yLabel} price expected to FALL`; pL.style.color='#ef4444';
    }
    pD.textContent='30s: '+(mid>0?'+':'')+mid.toFixed(3)+' · 2min: '+(last>0?'+':'')+last.toFixed(3);
  }

  // Momentum (Head C) — lower hazard = stronger momentum
  const mm=document.getElementById('momM'), ml=document.getElementById('momL');
  mm.innerHTML='';
  let avgHaz=0;
  for(let i=0;i<Math.min(hz.length,10);i++){
    const h=hz[i]||0; avgHaz+=h;
    const bar=document.createElement('div');
    bar.className='mom-bar';
    const pct=Math.max(10,(1-h)*100);
    bar.style.height=pct+'%';
    bar.style.background=h>0.75?'#ef4444':h>0.5?'#eab308':'#06b6d4';
    mm.appendChild(bar);
  }
  avgHaz=hz.length?avgHaz/hz.length:0;
  if(avgHaz<0.3){ml.textContent='STRONG';ml.style.color='#06b6d4'}
  else if(avgHaz<0.6){ml.textContent='MODERATE';ml.style.color='#eab308'}
  else{ml.textContent='FADING';ml.style.color='#ef4444'}

  // Market
  document.getElementById('mBid').textContent=(d.yes_bid||0)+'¢';
  document.getElementById('mAsk').textContent=(d.yes_ask||0)+'¢';
  document.getElementById('mSpr').textContent=((d.yes_ask||0)-(d.yes_bid||0))+'¢';
  const sd=f.score_diff||0;
  const sdE=document.getElementById('mSD');
  sdE.textContent=(sd>=0?'+':'')+sd.toFixed(0);
  sdE.style.color=sd>0?'#22c55e':sd<0?'#ef4444':'#e5e7eb';
  const mR=document.getElementById('mRun');
  if(rl>0){const who=rt>0?homeTeam:awayTeam;mR.textContent=who+' '+rpts+'pts';mR.style.color=rt>0?'#f97316':'#3b82f6'}
  else{mR.textContent='None';mR.style.color='#9ca3af'}

  // Badges
  document.getElementById('bPoss').textContent='POSS: '+(d.possession_id||'?');
  document.getElementById('bPipe').textContent=(d.pipeline_ms||0)+'ms';

  // Log
  const ll=document.createElement('div');ll.className='ll';
  const ts=new Date().toLocaleTimeString();
  const rc=rp>=.10?'color:#22c55e':'color:#9ca3af';
  const tc=tr[9]>0?'color:#22c55e':tr[9]<0?'color:#ef4444':'color:#9ca3af';
  ll.innerHTML='<span style="color:#6b7280">'+ts+'</span> '+
    'Run:<span style="'+rc+';font-weight:600"> '+(rp*100).toFixed(1)+'%</span> · '+
    'Price:<span style="'+tc+'"> '+(tr[9]>0?'↑':'↓')+Math.abs(tr[9]||0).toFixed(3)+'</span> · '+
    'Bid:'+(d.yes_bid||0)+'¢';
  const log=document.getElementById('log');
  log.appendChild(ll);log.scrollTop=log.scrollHeight;

  if(rp>=.10){sigs++;document.getElementById('tSig').textContent=sigs}
}
</script>
</body>
</html>
"""

