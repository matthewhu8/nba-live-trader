// Tests for the config-forwarding migration: the garbage/blowout gate
// thresholds and model paths moved from hardcoded Python into trading.yaml,
// loaded here and forwarded to the inference service. These tests guard that
// the new YAML keys parse and that the forwarded request bodies carry them.
package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTempConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "trading.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}

func TestLoadConfigParsesForwardedFields(t *testing.T) {
	path := writeTempConfig(t, `
inference:
  base_url: "http://localhost:8001"
  model_path: "models/saved/custom.pt"
  scaler_path: "models/saved/custom.pkl"
agent:
  min_yes_bid: 30
  max_yes_bid: 70
  min_abs_traj_entry: 0.12
  min_run_length_entry: 2
  blowout_margin_pts: 25
  garbage_time_period: 3
  garbage_time_clock_secs: 300
`)
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Inference.ModelPath != "models/saved/custom.pt" {
		t.Errorf("ModelPath = %q, want custom.pt", cfg.Inference.ModelPath)
	}
	if cfg.Inference.ScalerPath != "models/saved/custom.pkl" {
		t.Errorf("ScalerPath = %q, want custom.pkl", cfg.Inference.ScalerPath)
	}
	if cfg.Agent.BlowoutMarginPts != 25 {
		t.Errorf("BlowoutMarginPts = %d, want 25", cfg.Agent.BlowoutMarginPts)
	}
	if cfg.Agent.GarbageTimePeriod != 3 {
		t.Errorf("GarbageTimePeriod = %d, want 3", cfg.Agent.GarbageTimePeriod)
	}
	if cfg.Agent.GarbageTimeClockSecs != 300 {
		t.Errorf("GarbageTimeClockSecs = %d, want 300", cfg.Agent.GarbageTimeClockSecs)
	}
}

// Omitted keys must be the zero value so callers can detect "unset" and Python
// can fall back to its defaults (the omitempty wire contract depends on this).
func TestLoadConfigOmittedForwardedFieldsAreZero(t *testing.T) {
	path := writeTempConfig(t, "agent:\n  min_yes_bid: 30\n")
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Agent.BlowoutMarginPts != 0 || cfg.Agent.GarbageTimePeriod != 0 || cfg.Agent.GarbageTimeClockSecs != 0 {
		t.Errorf("omitted gate thresholds should be zero, got %d/%d/%d",
			cfg.Agent.BlowoutMarginPts, cfg.Agent.GarbageTimePeriod, cfg.Agent.GarbageTimeClockSecs)
	}
	if cfg.Inference.ModelPath != "" || cfg.Inference.ScalerPath != "" {
		t.Errorf("omitted model paths should be empty, got %q/%q", cfg.Inference.ModelPath, cfg.Inference.ScalerPath)
	}
}

func TestPossessionRequestCarriesGateThresholds(t *testing.T) {
	c := NewInferenceClient("http://x", InferenceConfig{
		BlowoutMarginPts:     25,
		GarbageTimePeriod:    3,
		GarbageTimeClockSecs: 300,
	})
	req := PossessionRequest{
		BlowoutMarginPts:     c.cfg.BlowoutMarginPts,
		GarbageTimePeriod:    c.cfg.GarbageTimePeriod,
		GarbageTimeClockSecs: c.cfg.GarbageTimeClockSecs,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{`"blowout_margin_pts":25`, `"garbage_time_period":3`, `"garbage_time_clock_secs":300`} {
		if !strings.Contains(string(data), want) {
			t.Errorf("possession body missing %s\ngot: %s", want, data)
		}
	}
}

// Zero-valued forwarded fields must be omitted so Python's defaults win.
func TestPossessionRequestOmitsZeroThresholds(t *testing.T) {
	data, err := json.Marshal(PossessionRequest{})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, absent := range []string{"blowout_margin_pts", "garbage_time_period", "garbage_time_clock_secs"} {
		if strings.Contains(string(data), absent) {
			t.Errorf("zero-value body should omit %s\ngot: %s", absent, data)
		}
	}
}
