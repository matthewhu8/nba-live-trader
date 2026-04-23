// Coordinator manages the lifecycle of all active GameEngines.
// Responsibilities:
//   - Determine which games are active tonight (Kalshi schedule or NBA scoreboard)
//   - Spawn one GameEngine goroutine per active game
//   - Restart engines that panic; cancel engines when games end
//   - Own the global RiskLedger (shared pointer passed to every engine)
//   - Own the KillSwitch (atomic bool checked by every goroutine)
package main

import (
	"context"
	"sync"

)

type Coordinator struct {
	ledger     *Ledger
	killSwitch *KillSwitch
	engines    map[string]*GameEngine
	mu         sync.Mutex
}

func NewCoordinator(ledger *Ledger, ks *KillSwitch) *Coordinator {
	return &Coordinator{
		ledger:     ledger,
		killSwitch: ks,
		engines:    make(map[string]*GameEngine),
	}
}

// Run blocks until ctx is cancelled. Polls for active games and manages engine lifecycle.
func (c *Coordinator) Run(ctx context.Context) error {
	// TODO: poll game schedule every 30s
	// TODO: spawn engine for each newly active game
	// TODO: cancel engine context when game ends (final buzzer detected)
	return nil
}

func (c *Coordinator) spawnEngine(ctx context.Context, gameID string) {
	// TODO: create GameEngine, start goroutine, register in c.engines
}
