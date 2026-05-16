package main

import "testing"

func newTestLedger(cfg RiskConfig) (*Ledger, *KillSwitch) {
	ks := NewKillSwitch()
	return NewLedger(cfg, ks), ks
}

func defaultRiskCfg() RiskConfig {
	return RiskConfig{
		MaxTotalExposureCents:   5000,
		MaxPerGameExposureCents: 2000,
		MaxDailyLossCents:       10000,
		MaxContractsPerOrder:    100,
	}
}

func TestCheck(t *testing.T) {
	cases := []struct {
		name        string
		setup       func(*Ledger, *KillSwitch)
		action      string
		gameID      string
		yesBid      int
		contracts   int
		wantOK      bool
		wantKillSet bool
	}{
		{
			name:      "kill_switch_already_set",
			setup:     func(_ *Ledger, ks *KillSwitch) { ks.Set() },
			action:    "BUY_YES", gameID: "g1", yesBid: 50, contracts: 10,
			wantOK: false,
		},
		{
			name:      "wait_action_always_approved",
			action:    "WAIT", gameID: "g1", yesBid: 50, contracts: 0,
			wantOK: true,
		},
		{
			name:      "exit_action_always_approved",
			action:    "EXIT", gameID: "g1", yesBid: 50, contracts: 0,
			wantOK: true,
		},
		{
			name:      "order_size_cap_exceeded",
			action:    "BUY_YES", gameID: "g1", yesBid: 50, contracts: 101,
			wantOK: false,
		},
		{
			name: "daily_loss_at_limit_trips_kill_switch",
			setup: func(l *Ledger, _ *KillSwitch) {
				// Put daily P&L just past the limit so Check() sees the breach.
				l.dailyPnL = -(defaultRiskCfg().MaxDailyLossCents + 1)
			},
			action:      "BUY_YES", gameID: "g1", yesBid: 50, contracts: 10,
			wantOK:      false,
			wantKillSet: true,
		},
		{
			name: "per_game_exposure_would_exceed",
			setup: func(l *Ledger, _ *KillSwitch) {
				// Pre-fill per-game exposure so the next order would push it over.
				// MaxPerGameExposureCents=2000; 39×50=1950 in, next 10×50=500 → 2450 > 2000
				l.perGameExposure["g1"] = 1950
				l.totalExposure = 1950
			},
			action:    "BUY_YES", gameID: "g1", yesBid: 50, contracts: 10,
			wantOK: false,
		},
		{
			name: "total_exposure_would_exceed",
			setup: func(l *Ledger, _ *KillSwitch) {
				// Fill total exposure near the cap but across different games.
				l.totalExposure = 4800
				l.perGameExposure["g2"] = 4800
			},
			action:    "BUY_YES", gameID: "g1", yesBid: 50, contracts: 10,
			wantOK: false,
		},
		{
			name:      "happy_path_nothing_at_limits",
			action:    "BUY_YES", gameID: "g1", yesBid: 50, contracts: 10,
			wantOK: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l, ks := newTestLedger(defaultRiskCfg())
			if tc.setup != nil {
				tc.setup(l, ks)
			}
			ok, _ := l.Check(tc.action, tc.gameID, tc.yesBid, tc.contracts)
			if ok != tc.wantOK {
				t.Errorf("Check() approved=%v, want %v", ok, tc.wantOK)
			}
			if tc.wantKillSet && !ks.IsSet() {
				t.Errorf("expected kill switch to be set, but it is not")
			}
		})
	}
}

func TestRecordFill(t *testing.T) {
	t.Run("increments_total_and_per_game_exposure", func(t *testing.T) {
		l, _ := newTestLedger(defaultRiskCfg())
		l.RecordFill("g1", 100, 50)

		if l.totalExposure != 5000 {
			t.Errorf("totalExposure=%d, want 5000", l.totalExposure)
		}
		if l.perGameExposure["g1"] != 5000 {
			t.Errorf("perGameExposure[g1]=%d, want 5000", l.perGameExposure["g1"])
		}
	})

	t.Run("two_games_tracked_independently", func(t *testing.T) {
		l, _ := newTestLedger(defaultRiskCfg())
		l.RecordFill("g1", 10, 50)  // 500 cents
		l.RecordFill("g2", 20, 40)  // 800 cents

		if l.totalExposure != 1300 {
			t.Errorf("totalExposure=%d, want 1300", l.totalExposure)
		}
		if l.perGameExposure["g1"] != 500 {
			t.Errorf("perGameExposure[g1]=%d, want 500", l.perGameExposure["g1"])
		}
		if l.perGameExposure["g2"] != 800 {
			t.Errorf("perGameExposure[g2]=%d, want 800", l.perGameExposure["g2"])
		}
	})
}

func TestRecordExit(t *testing.T) {
	t.Run("decrements_exposure_and_updates_pnl", func(t *testing.T) {
		l, _ := newTestLedger(defaultRiskCfg())
		l.RecordFill("g1", 100, 50)   // buy 100 @ 50¢ → exposure=5000
		l.RecordExit("g1", 100, 50, 52) // exit @ 52¢ → pnl=+200

		if l.totalExposure != 0 {
			t.Errorf("totalExposure=%d, want 0", l.totalExposure)
		}
		if l.perGameExposure["g1"] != 0 {
			t.Errorf("perGameExposure[g1]=%d, want 0", l.perGameExposure["g1"])
		}
		if l.dailyPnL != 200 {
			t.Errorf("dailyPnL=%d, want 200", l.dailyPnL)
		}
	})

	t.Run("clamps_exposure_to_zero_on_double_exit", func(t *testing.T) {
		l, _ := newTestLedger(defaultRiskCfg())
		l.RecordFill("g1", 10, 50)
		l.RecordExit("g1", 10, 50, 52)
		// Double-exit (simulates a bug where exit is called twice)
		l.RecordExit("g1", 10, 50, 52)

		if l.totalExposure < 0 {
			t.Errorf("totalExposure went negative: %d", l.totalExposure)
		}
		if l.perGameExposure["g1"] < 0 {
			t.Errorf("perGameExposure[g1] went negative: %d", l.perGameExposure["g1"])
		}
	})

	t.Run("trips_kill_switch_on_cumulative_daily_loss", func(t *testing.T) {
		cfg := defaultRiskCfg()
		cfg.MaxDailyLossCents = 1000
		l, ks := newTestLedger(cfg)

		// Two losing positions that together exceed the daily loss limit.
		l.RecordFill("g1", 100, 50)
		l.RecordExit("g1", 100, 50, 44) // lose 600¢

		if ks.IsSet() {
			t.Errorf("kill switch tripped too early (after first exit, pnl=-600)")
		}

		l.RecordFill("g1", 100, 50)
		l.RecordExit("g1", 100, 50, 44) // lose another 600¢ → total -1200 > -1000

		if !ks.IsSet() {
			t.Errorf("expected kill switch to be set after cumulative loss -1200 > limit -1000")
		}
	})
}
