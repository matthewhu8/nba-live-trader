// Entry point for the live trading engine.
// For now: test mode polls a single game's NBA feed and prints events.
//
// Usage:
//
//	go run . --game 0022501234           poll a specific game ID (Ctrl+C to stop)
//	go run . --game 0022501234 --test    run for 30s then exit
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// currently set to only run nba live feed to ensure we are processing
// each possession correctly (for dev purposes)
func main() {
	// process inputs to determine mode
	gameID := flag.String("game", "", "NBA game ID to poll (e.g. 0022501234)")
	testMode := flag.Bool("test", false, "run for 30s then exit")
	flag.Parse()

	if *gameID == "" {
		log.Fatal("usage: go run . --game <game_id>")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if *testMode {
		var tc context.CancelFunc
		ctx, tc = context.WithTimeout(ctx, 30*time.Second)
		defer tc()
		log.Printf("test mode: polling game=%s for 30s", *gameID)
	}

	events := make(chan NBAEvent, 500)
	feed := NewNBAFeed(*gameID)
	go feed.Run(ctx, events) // runs NBAFeed in a goroutine and continuously polls for new events, outputting to events channel

	for {
		select {
		case ev := <-events:
			log.Printf("EVENT action=%d type=%-15s period=%d clock=%s home=%s away=%s desc=%q",
				ev.ActionNumber, ev.ActionType, ev.Period, ev.Clock,
				ev.ScoreHome, ev.ScoreAway, ev.Description)
		case <-ctx.Done():
			log.Println("shutting down")
			return
		}
	}
}
