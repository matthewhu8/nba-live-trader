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
func (l *Ledger) Check(action, gameID string, yesBid int) (bool, string) {
	if l.killSwitch.IsSet() {
		return false, "kill switch active"
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if action == "WAIT" || action == "EXIT" {
		return true, ""
	}

	// TODO: estimate exposure for this order (contracts × price)
	// TODO: check daily loss limit
	// TODO: check per-game exposure
	// TODO: check total exposure
	return true, ""
}

// RecordFill updates exposure and PnL after a confirmed order fill.
func (l *Ledger) RecordFill(gameID string, contracts, entryPrice int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	// TODO
}

// RecordExit updates PnL and reduces open exposure on position close.
func (l *Ledger) RecordExit(gameID string, contracts, exitPrice, entryPrice int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	// TODO: pnl = (exitPrice - entryPrice) * contracts - fees
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
