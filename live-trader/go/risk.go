// Ledger is shared by pointer across every GameEngine, so its limits are global
// rather than per-game. All operations are mutex-protected. Every order must pass
// Check before it reaches the Router.
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
	totalExposure   int            // cents open across all games
	perGameExposure map[string]int // game_id to open exposure in cents
	dailyPnL        int            // cents; negative is a loss
	killSwitch      *KillSwitch
}

func NewLedger(cfg RiskConfig, ks *KillSwitch) *Ledger {
	return &Ledger{
		cfg:             cfg,
		perGameExposure: make(map[string]int),
		killSwitch:      ks,
	}
}

// Check returns (approved, reason) and must be called synchronously before every
// order. yesBid is a conservative price estimate for both sides, since a NO entry
// costs at most the yes bid.
func (l *Ledger) Check(action, gameID string, yesBid, contracts int) (bool, string) {
	if l.killSwitch.IsSet() {
		return false, "kill switch active"
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if action == "WAIT" || action == "EXIT" {
		return true, ""
	}

	if contracts > l.cfg.MaxContractsPerOrder {
		return false, fmt.Sprintf("order size %d exceeds max %d contracts", contracts, l.cfg.MaxContractsPerOrder)
	}

	// Trip the kill switch on a daily loss breach so every engine stops at once.
	if l.dailyPnL < -l.cfg.MaxDailyLossCents {
		l.killSwitch.Set()
		return false, fmt.Sprintf("daily loss limit: pnl=%dc limit=%dc, kill switch activated", l.dailyPnL, l.cfg.MaxDailyLossCents)
	}

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

// RecordFill updates open exposure after a confirmed fill.
func (l *Ledger) RecordFill(gameID string, contracts, entryPrice int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	exposure := contracts * entryPrice
	l.totalExposure += exposure
	l.perGameExposure[gameID] += exposure
}

// RecordExit reduces open exposure and updates daily P&L when a position closes,
// tripping the kill switch if that puts us past the daily loss limit.
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
	// Gross of fees. The Router accounts for those separately.
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

// KillSwitch is an atomic bool shared by every goroutine. Once set, each GameEngine
// exits within one loop iteration and no further orders are placed.
type KillSwitch struct {
	val atomic.Bool
}

func NewKillSwitch() *KillSwitch   { return &KillSwitch{} }
func (ks *KillSwitch) Set()        { ks.val.Store(true) }
func (ks *KillSwitch) IsSet() bool { return ks.val.Load() }
