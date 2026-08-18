// JSONLogger writes one JSON object per line to live-trader.jsonl. Every Emit takes
// a mutex, so concurrent game engines cannot interleave bytes inside a record.
//
// Writes are best-effort: errors warn at most once a minute and are then dropped,
// and Emit on a nil receiver is a no-op, so no call site has to nil-check. A
// logging failure never breaks the trading loop.
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

// NewJSONLogger opens live-trader.jsonl in the run dir, creating or appending. On
// failure the caller should warn and carry on with a nil logger.
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

// Emit writes one record. gameID may be empty for run-level events. The envelope
// fields (schema_version, ts, run_id, event, game_id) are added here and override
// any same-named key in fields.
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
	// Envelope goes last so it wins any collision with a caller-supplied key.
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

// Close flushes and closes the file. Idempotent and safe to call on nil.
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
