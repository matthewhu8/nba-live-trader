// Entry point for the live trading engine.
// Reads config, starts the Python inference service health check,
// then hands off to the Coordinator.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	_ = ctx
	// TODO: load config/trading.yaml
	// TODO: start coordinator
}
