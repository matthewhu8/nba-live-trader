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
)

// currently set to only run nba live feed to ensure we are processing
// each possession correctly (for dev purposes)
func main() {
	// process inputs to determine mode
	gameID := flag.String("game", "", "NBA game ID to poll (e.g. 0022501234)")
	marketTicker := flag.String("market", "", "Kalshi market ticker (e.g. NBA_Game_20260423_LALHOU)") // returns address of string
	flag.Parse()

	if *gameID == "" {
		log.Fatal("usage: go run . --game <game_id>")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// set up channels for streaming data from the NBA feed and Kalshi feed
	events := make(chan NBAEvent, 500) // channel for NBA events (possessions)
	nbaFeed := NewNBAFeed(*gameID)
	go nbaFeed.Run(ctx, events) // runs NBAFeed in a goroutine and continuously polls for new events, outputting to events channel

	ticks := make(chan KalshiTick, 50000) // channel for Kalshi ticks
	kalshiFeed := NewKalshiFeed(*marketTicker, "") // marketTicker is the address of the string, so *marketTicker is the value
	go kalshiFeed.Run(ctx, ticks)
	
	for {
		select {
		case ev := <-events:
			log.Printf("NBA EVENT action=%d type=%-15s period=%d clock=%s home=%s away=%s desc=%q",
				ev.ActionNumber, ev.ActionType, ev.Period, ev.Clock,
				ev.ScoreHome, ev.ScoreAway, ev.Description)
		case tick := <-ticks:
			log.Printf("TICK: yes_bid=%d yes_ask=%d yes_last=%d volume=%d open_interest=%d is_stale=%t",
				tick.YesBid, tick.YesAsk, tick.YesLast, tick.Volume, tick.OpenInterest, tick.IsStale)
		case <-ctx.Done():
			log.Println("shutting down")
			return
		}
	}
}
