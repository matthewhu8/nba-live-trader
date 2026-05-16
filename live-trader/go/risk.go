// RiskLedger is shared across all GameEngines via pointer.
// Every order must pass Check() before reaching the OrderRouter.
// All operations are mutex-protected — safe for concurrent game goroutines.
//
// Hard limits (loaded from config/trading.yaml):
//   max_total_exposure_cents    — max open exposure across all games combined
//   max_per_game_exposure_cents — max open exposure for a single game
//   max_daily_loss_cents        — stop trading for the day if exceeded
//   max_contracts_per_order     — single order size cap
package main

import (
	"fmt"
	"sync"
	"sync/atomic"
)

type RiskConfig struct {
	MaxTotalExposureCents   int
	MaxPerGameExposureCents int
	MaxDailyLossCents       int
	MaxContractsPerOrder    int
}

type Ledger struct {
	mu              sync.Mutex
	cfg             RiskConfig
	totalExposure   int            // cents, current open across all games
	perGameExposure map[string]int // game_id → open exposure in cents
	dailyPnL        int            // cents, negative = loss
	killSwitch      *KillSwitch
}

func NewLedger(cfg RiskConfig, ks *KillSwitch) *Ledger {
	return &Ledger{
		cfg:             cfg,
		perGameExposure: make(map[string]int),
		killSwitch:      ks,
	}
}

// Check returns (approved, reason). Reason is non-empty only when rejected.
// Must be called synchronously before every order.
// contracts is the intended order size; yesBid is used as a conservative
// price estimate for both YES and NO orders (NO entry price ≤ yesBid).
func (l *Ledger) Check(action, gameID string, yesBid, contracts int) (bool, string) {
	if l.killSwitch.IsSet() {
		return false, "kill switch active"
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if action == "WAIT" || action == "EXIT" {
		return true, ""
	}

	// Single-order size cap.
	if contracts > l.cfg.MaxContractsPerOrder {
		return false, fmt.Sprintf("order size %d exceeds max %d contracts", contracts, l.cfg.MaxContractsPerOrder)
	}

	// Daily loss halt — trip kill switch so all engines stop immediately.
	if l.dailyPnL < -l.cfg.MaxDailyLossCents {
		l.killSwitch.Set()
		return false, fmt.Sprintf("daily loss limit: pnl=%dc limit=%dc — kill switch activated", l.dailyPnL, l.cfg.MaxDailyLossCents)
	}

	// Exposure estimate: contracts × entry price in cents.
	// yesBid is a conservative upper bound (NO orders cost 100 - ask ≤ bid).
	newExposure := contracts * yesBid

	if l.perGameExposure[gameID]+newExposure > l.cfg.MaxPerGameExposureCents {
		return false, fmt.Sprintf("per-game exposure %dc + %dc would exceed limit %dc",
			l.perGameExposure[gameID], newExposure, l.cfg.MaxPerGameExposureCents)
	}

	if l.totalExposure+newExposure > l.cfg.MaxTotalExposureCents {
		return false, fmt.Sprintf("total exposure %dc + %dc would exceed limit %dc",
			l.totalExposure, newExposure, l.cfg.MaxTotalExposureCents)
	}

	return true, ""
}

// RecordFill updates open exposure after a confirmed paper fill.
func (l *Ledger) RecordFill(gameID string, contracts, entryPrice int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	exposure := contracts * entryPrice
	l.totalExposure += exposure
	l.perGameExposure[gameID] += exposure
}

// RecordExit reduces open exposure and updates daily P&L on position close.
// Trips the kill switch if the daily loss limit is breached post-exit.
func (l *Ledger) RecordExit(gameID string, contracts, entryPrice, exitPrice int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	exposure := contracts * entryPrice
	l.totalExposure -= exposure
	if l.totalExposure < 0 {
		l.totalExposure = 0
	}
	l.perGameExposure[gameID] -= exposure
	if l.perGameExposure[gameID] < 0 {
		l.perGameExposure[gameID] = 0
	}
	// Gross P&L in cents (fee deduction is logged separately by OrderRouter).
	l.dailyPnL += (exitPrice - entryPrice) * contracts
	if l.dailyPnL < -l.cfg.MaxDailyLossCents {
		l.killSwitch.Set()
	}
}

func (l *Ledger) Summary() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return fmt.Sprintf("exposure=%dc dailyPnL=%dc", l.totalExposure, l.dailyPnL)
}

// KillSwitch is an atomic bool shared by every goroutine.
// When set, all GameEngines exit within one loop iteration and no new orders are placed.
// Triggered by: daily loss breach, manual signal, or config flag at startup.
type KillSwitch struct {
	val atomic.Bool
}

func NewKillSwitch() *KillSwitch { return &KillSwitch{} }
func (ks *KillSwitch) Set()      { ks.val.Store(true) }
func (ks *KillSwitch) IsSet() bool { return ks.val.Load() }
