// JSONLogger writes one JSON object per line to live-trader.jsonl in the
// run directory. Goroutine-safe: every Emit acquires a mutex so concurrent
// game engines from the Coordinator do not interleave bytes within a record.
//
// Best-effort semantics: write errors are warned (rate-limited to one per
// minute) and silently dropped. The trading loop never fails because of a
// logging error. Calling Emit on a nil receiver is a deliberate no-op so
// callers don't have to nil-check every emit site.
//
// Phase 1 emits only run_start / run_end. Phase 2+ adds the rich event set
// (possession, entry, hold, exit, market_swap, etc.).
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type JSONLogger struct {
	mu       sync.Mutex
	file     *os.File
	runID    string
	closed   bool
	lastWarn time.Time
}

// NewJSONLogger opens (or creates+appends) live-trader.jsonl in the run dir.
// On open failure, returns nil + error — caller should warn and proceed with
// a nil logger (Emit on nil is a safe no-op).
func NewJSONLogger(run *Run) (*JSONLogger, error) {
	if run == nil {
		return nil, fmt.Errorf("nil run")
	}
	path := filepath.Join(run.LogDir, "live-trader.jsonl")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open jsonl %s: %w", path, err)
	}
	return &JSONLogger{file: f, runID: run.ID}, nil
}

// Emit writes a single record. event is the discriminator (e.g. "run_start",
// "possession"); gameID may be empty for run-level events; fields carries the
// event-specific payload.
//
// Envelope fields (schema_version, ts, run_id, event, game_id) are added
// automatically and override any same-named keys in fields.
func (l *JSONLogger) Emit(event, gameID string, fields map[string]interface{}) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}

	record := make(map[string]interface{}, 5+len(fields))
	for k, v := range fields {
		record[k] = v
	}
	// Envelope last so it wins over any caller-supplied collision.
	record["schema_version"] = 1
	record["ts"] = time.Now().UTC().Format("2006-01-02T15:04:05.000Z")
	record["run_id"] = l.runID
	record["event"] = event
	if gameID != "" {
		record["game_id"] = gameID
	}

	encoded, err := json.Marshal(record)
	if err != nil {
		l.warnRateLimited("marshal failed", err)
		return
	}
	encoded = append(encoded, '\n')
	if _, err := l.file.Write(encoded); err != nil {
		l.warnRateLimited("write failed", err)
	}
}

// Close flushes and closes the underlying file. Safe to call on nil and
// idempotent.
func (l *JSONLogger) Close() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	l.closed = true
	_ = l.file.Close()
}

func (l *JSONLogger) warnRateLimited(msg string, err error) {
	now := time.Now()
	if now.Sub(l.lastWarn) < time.Minute {
		return
	}
	l.lastWarn = now
	zlog.Warn().Err(err).Msg("jsonlog: " + msg)
}
