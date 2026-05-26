// A Run identifies a single Go process invocation. The Coordinator may manage
// many games inside one Run, but they all share the same RunID.
//
// Each Run gets a dedicated directory under logs/runs/{date}/{run_id}/ which
// will hold (across phases):
//   - manifest.json     config snapshot, git SHA, paper/live, started/ended
//   - live-trader.jsonl one JSON object per event (Phase 2+)
//   - stderr.log        captured zerolog warnings/errors (Phase 7)
//
// Run is constructed once in main.go and threaded through Coordinator and
// every GameEngine. It is intentionally lightweight — no I/O happens during
// Phase 1 inside the trading loop, only at startup and shutdown.
package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

type Run struct {
	ID        string
	StartedAt time.Time
	LogDir    string
	PaperMode bool

	mu           sync.Mutex
	manifestPath string
}

// NewRun creates the run directory and returns a Run handle. LogDir is
// stored as an absolute path so the Python inference service (which runs
// from a different working directory) can find the same directory when we
// pass log_dir on /game/start. Does not yet write the manifest — call
// WriteManifest after the full Config is in hand.
func NewRun(paperMode bool) (*Run, error) {
	id := generateRunID()
	startedAt := time.Now().UTC()
	date := startedAt.Format("2006-01-02")
	logDir := filepath.Join("logs", "runs", date, id)

	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, fmt.Errorf("create run log dir %s: %w", logDir, err)
	}

	absLogDir, err := filepath.Abs(logDir)
	if err != nil {
		// Fallback to relative — manifest still works, only Python plumbing
		// might fail to resolve (in which case Python falls back to no JSONL).
		absLogDir = logDir
	}

	return &Run{
		ID:           id,
		StartedAt:    startedAt,
		LogDir:       absLogDir,
		PaperMode:    paperMode,
		manifestPath: filepath.Join(absLogDir, "manifest.json"),
	}, nil
}

// generateRunID returns "20260509-034211-9a7b3c" — sortable, unique enough
// across overlapping process starts (3 random bytes = 16M space per second).
func generateRunID() string {
	ts := time.Now().UTC().Format("20060102-150405")
	b := make([]byte, 3)
	if _, err := rand.Read(b); err != nil {
		// Crypto rand failure is exotic; fall back to something deterministic-ish.
		return ts + "-000000"
	}
	return fmt.Sprintf("%s-%s", ts, hex.EncodeToString(b))
}

// WriteManifest writes the initial manifest.json. Best-effort: failures log
// a warning but do not propagate, so a missing/unwritable log dir cannot
// crash the engine.
func (r *Run) WriteManifest(cfg Config) {
	if r == nil {
		return
	}
	manifest := map[string]interface{}{
		"schema_version": 1,
		"run_id":         r.ID,
		"started_at":     r.StartedAt.Format(time.RFC3339),
		"ended_at":       nil,
		"end_reason":     nil,
		"pid":            os.Getpid(),
		"git_sha":        readGitSHA(),
		"paper_mode":     r.PaperMode,
		"config":         cfg,
		"env_present":    detectEnvPresent(),
	}
	if err := writeJSONFile(r.manifestPath, manifest); err != nil {
		zlog.Warn().Err(err).Str("path", r.manifestPath).Msg("manifest write failed — continuing")
	}
}

// FinalizeManifest reopens the manifest, sets ended_at + end_reason +
// summary stats, and rewrites it. Best-effort. Safe to call multiple times
// (last call wins). Summary is computed from live-trader.jsonl so callers
// don't need to thread shared counters across game engines.
func (r *Run) FinalizeManifest(endReason string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	data, err := os.ReadFile(r.manifestPath)
	if err != nil {
		zlog.Warn().Err(err).Msg("finalize manifest: read failed")
		return
	}
	var manifest map[string]interface{}
	if err := json.Unmarshal(data, &manifest); err != nil {
		zlog.Warn().Err(err).Msg("finalize manifest: parse failed")
		return
	}
	endedAt := time.Now().UTC()
	manifest["ended_at"] = endedAt.Format(time.RFC3339)
	manifest["end_reason"] = endReason
	manifest["summary"] = r.computeSummary(endedAt)
	if err := writeJSONFile(r.manifestPath, manifest); err != nil {
		zlog.Warn().Err(err).Msg("finalize manifest: write failed")
	}
}

// CaptureStderr redirects zerolog warnings/errors to stderr.log in the run
// directory while still printing them to the terminal (io.MultiWriter).
// Returns the file handle so main can defer-close it. Best-effort: any
// failure returns nil and leaves zlog targeting stderr only.
func (r *Run) CaptureStderr() *os.File {
	if r == nil {
		return nil
	}
	path := filepath.Join(r.LogDir, "stderr.log")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		log.Printf("[WARN] could not open stderr.log: %v — zerolog stays on stderr only", err)
		return nil
	}
	// Replace the package-level zlog writer. All future zlog calls land
	// in both terminal and file. Existing call sites are unchanged.
	zlog = zerolog.New(io.MultiWriter(os.Stderr, f)).With().Timestamp().Logger()
	return f
}

// computeSummary scans live-trader.jsonl and aggregates per-event counters.
// Lives here (not in jsonlog.go) because it's a manifest concern: nothing
// in the trading loop needs this data. Returns empty stats if the JSONL
// file is missing or unreadable — the manifest still gets ended_at.
func (r *Run) computeSummary(endedAt time.Time) map[string]interface{} {
	durationSecs := int(endedAt.Sub(r.StartedAt).Seconds())

	jsonlPath := filepath.Join(r.LogDir, "live-trader.jsonl")
	f, err := os.Open(jsonlPath)
	if err != nil {
		return map[string]interface{}{
			"duration_secs": durationSecs,
			"jsonl_error":   err.Error(),
		}
	}
	defer f.Close()

	var (
		gamesRun         int
		totalPossessions int
		backfillSeen     int
		tradesOpened     int
		tradesClosed     int
		wins             int
		netPnL           float64
		errorEvents      int
		garbageEvents    int
		clientMSSum      int64
		clientMSCount    int64
	)

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<24) // up to 16MB lines (defensive)
	for scanner.Scan() {
		var rec map[string]interface{}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue // best-effort: skip malformed lines
		}
		event, _ := rec["event"].(string)
		switch event {
		case "game_start":
			gamesRun++
		case "possession":
			isBackfill, _ := rec["is_backfill"].(bool)
			if isBackfill {
				backfillSeen++
				continue
			}
			totalPossessions++
			if v, ok := rec["client_total_ms"].(float64); ok {
				clientMSSum += int64(v)
				clientMSCount++
			}
		case "entry":
			tradesOpened++
		case "exit":
			tradesClosed++
			if v, ok := rec["net_pnl_dollars"].(float64); ok {
				netPnL += v
				if v > 0 {
					wins++
				}
			}
		case "error":
			errorEvents++
		case "garbage_time":
			garbageEvents++
		}
	}

	avgClientMS := float64(0)
	if clientMSCount > 0 {
		avgClientMS = float64(clientMSSum) / float64(clientMSCount)
	}

	return map[string]interface{}{
		"duration_secs":     durationSecs,
		"games_run":         gamesRun,
		"total_possessions": totalPossessions,
		"backfill_seen":     backfillSeen,
		"trades_opened":     tradesOpened,
		"trades_closed":     tradesClosed,
		"wins":              wins,
		"net_pnl_dollars":   netPnL,
		"errors":            errorEvents,
		"garbage_events":    garbageEvents,
		"avg_client_ms":     avgClientMS,
	}
}

// readGitSHA shells out to `git rev-parse HEAD`. Returns "unknown" on any
// failure (e.g. running outside a repo, git not installed).
func readGitSHA() string {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// detectEnvPresent reports which relevant env vars are set, by name only.
// Values are NEVER recorded — this is for repro audits, not credential
// inventory.
func detectEnvPresent() map[string]bool {
	keys := []string{
		"KALSHI_KEY_ID",
		"KALSHI_PEM_PATH",
		"MOTHERDUCK_TOKEN",
	}
	out := make(map[string]bool, len(keys))
	for _, k := range keys {
		out[k] = os.Getenv(k) != ""
	}
	return out
}

// writeJSONFile writes v as pretty JSON, atomically (temp file + rename).
func writeJSONFile(path string, v interface{}) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}
