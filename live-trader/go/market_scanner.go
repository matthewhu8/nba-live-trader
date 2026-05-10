// MarketScanner polls the Kalshi REST API for the markets tied to one event
// (e.g., all spreads under KXNBASPREAD-26MAY10NYKPHI) and hot-swaps the active
// WebSocket subscription to whichever market has a yes_bid closest to 50¢ —
// but ONLY when the currently-subscribed market drifts outside [minYesBid,
// maxYesBid].
//
// Phase 1 (observability) additions, 2026-05-10:
//   - `market_scan` JSONL event emitted on every poll (every 10s) so we can
//     answer "what was the scanner seeing at any moment" from disk alone.
//   - `market_swap` JSONL event emitted whenever an actual swap fires.
//   - log.Printf calls replaced with zlog so the lines also land in stderr.log
//     (Phase 7 io.MultiWriter capture).
//   - scanWithCurrent extended to return n_candidates so the JSONL can record
//     how many markets qualified the spread / liquidity filters on each poll.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"time"
)

type MarketScanner struct {
	eventTicker string
	minYesBid   int
	maxYesBid   int
	jsonLog     *JSONLogger // nil-safe — passed in from the game engine
	gameID      string      // for JSONL game_id field
}

func NewMarketScanner(eventTicker string, minYesBid, maxYesBid int, jsonLog *JSONLogger, gameID string) *MarketScanner {
	return &MarketScanner{
		eventTicker: eventTicker,
		minYesBid:   minYesBid,
		maxYesBid:   maxYesBid,
		jsonLog:     jsonLog,
		gameID:      gameID,
	}
}

// Run polls every 10 seconds and sends new market tickers on outCh whenever
// a swap is warranted. Returns immediately if eventTicker is empty.
func (s *MarketScanner) Run(ctx context.Context, initialTicker string, outCh chan<- string) {
	if s.eventTicker == "" {
		return
	}

	currentTicker := initialTicker
	currentBid := 50 // assumed mid until the first poll fills it in

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	// Initial scan — if best != current, swap immediately.
	if best, bestBid, _, nCandidates, err := s.scanWithCurrent(ctx, currentTicker); err == nil {
		if best != "" && best != currentTicker {
			s.emitSwap(currentTicker, best, currentBid, bestBid, "initial")
			zlog.Info().
				Str("new_market", best).
				Int("bid_cents", bestBid).
				Msg("market scanner: initial market selected")
			currentTicker = best
			currentBid = bestBid
			outCh <- best
		}
		// Emit one "initial" market_scan record so the JSONL has a baseline
		// of what the scanner saw at startup.
		s.emitScan(currentTicker, currentBid, best, bestBid, nCandidates, true, "initial")
	} else {
		zlog.Warn().Err(err).Msg("market scanner: initial poll failed")
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			best, bestBid, currentActiveBid, nCandidates, err := s.scanWithCurrent(ctx, currentTicker)
			if err != nil {
				zlog.Warn().Err(err).Msg("market scanner: poll failed")
				continue
			}

			currentBid = currentActiveBid
			inBand := currentBid >= s.minYesBid && currentBid <= s.maxYesBid

			// Decision tree — used for both the JSONL record and the swap.
			decision := "stay"
			if !inBand {
				switch {
				case best == "":
					decision = "no_candidates"
				case best == currentTicker:
					decision = "best_is_current"
				default:
					decision = "swap"
				}
			}

			s.emitScan(currentTicker, currentBid, best, bestBid, nCandidates, inBand, decision)

			if decision == "swap" {
				s.emitSwap(currentTicker, best, currentBid, bestBid, "out_of_band")
				zlog.Info().
					Str("old_market", currentTicker).
					Int("old_bid_cents", currentBid).
					Str("new_market", best).
					Int("new_bid_cents", bestBid).
					Msg("market scanner: swapping")
				currentTicker = best
				currentBid = bestBid

				select {
				case outCh <- best:
				default:
				}
			}
		}
	}
}

// emitScan writes one market_scan JSONL line. Called every 10s, regardless of
// whether a swap fires — that's the whole point: we want a continuous trace.
func (s *MarketScanner) emitScan(curT string, curBid int, bestT string, bestBid int, nCandidates int, inBand bool, decision string) {
	if s.jsonLog == nil {
		return
	}
	s.jsonLog.Emit("market_scan", s.gameID, map[string]interface{}{
		"current_ticker": curT,
		"current_bid":    curBid,
		"best_ticker":    bestT,
		"best_bid":       bestBid,
		"n_candidates":   nCandidates,
		"in_band":        inBand,
		"decision":       decision, // "initial" | "stay" | "swap" | "no_candidates" | "best_is_current"
	})
}

// emitSwap writes one market_swap JSONL line. Fires when the scanner actually
// pushes a new ticker onto outCh (or on the initial selection).
func (s *MarketScanner) emitSwap(oldT, newT string, oldBid, newBid int, reason string) {
	if s.jsonLog == nil {
		return
	}
	s.jsonLog.Emit("market_swap", s.gameID, map[string]interface{}{
		"old_ticker": oldT,
		"new_ticker": newT,
		"old_bid":    oldBid,
		"new_bid":    newBid,
		"reason":     reason, // "initial" | "out_of_band"
	})
}

// ─────────────────────────────────────────────────────────────────────────────

type kalshiMarketResp struct {
	Markets []struct {
		Ticker         string `json:"ticker"`
		YesBidDollars  string `json:"yes_bid_dollars"`
		YesAskDollars  string `json:"yes_ask_dollars"`
		VolumeFP       string `json:"volume_fp"`
		OpenInterestFP string `json:"open_interest_fp"`
		Status         string `json:"status"`
	} `json:"markets"`
}

// scanWithCurrent queries Kalshi /markets?event_ticker=… and returns:
//   - bestTicker:        the active market with bid closest to 50¢ that passes
//                        the liquidity + spread filters; "" if none qualify
//   - bestBid:           that market's yes_bid in cents (0 if no candidate)
//   - currentActiveBid:  the live yes_bid of currentTicker if it appears in
//                        the response and is active; defaults to 50¢ otherwise
//                        (the "didn't find it, assume mid" sentinel)
//   - nCandidates:       how many markets passed all filters this poll —
//                        useful for diagnosing "why was no swap proposed"
//   - err:               only HTTP/JSON errors
func (s *MarketScanner) scanWithCurrent(ctx context.Context, currentTicker string) (bestTicker string, bestBid int, currentActiveBid int, nCandidates int, err error) {
	requestURL := fmt.Sprintf("%s?event_ticker=%s", kalshiRESTURL("/markets"), url.QueryEscape(s.eventTicker))

	req, err := http.NewRequestWithContext(ctx, "GET", requestURL, nil)
	if err != nil {
		return "", 0, 0, 0, err
	}

	headers, err := GetKalshiAuthHeaders("GET", "/markets")
	if err != nil {
		return "", 0, 0, 0, err
	}
	for k, v := range headers {
		req.Header[k] = v
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", 0, 0, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, 0, fmt.Errorf("kalshi API returned %d", resp.StatusCode)
	}

	var data kalshiMarketResp
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", 0, 0, 0, err
	}

	type candidate struct {
		ticker string
		bid    int
		dist   int
	}

	var candidates []candidate
	currentActiveBid = 50 // sentinel — overwritten if currentTicker is found

	for _, m := range data.Markets {
		if m.Status != "active" {
			continue
		}

		yesBid := dollarsStrToCents(m.YesBidDollars)
		yesAsk := dollarsStrToCents(m.YesAskDollars)

		if m.Ticker == currentTicker {
			currentActiveBid = yesBid
		}

		if yesBid == 0 || yesAsk == 0 {
			continue // no liquidity
		}

		spread := yesAsk - yesBid
		if spread > 10 || spread < 0 {
			continue // spread too wide or inverted
		}

		dist := yesBid - 50
		if dist < 0 {
			dist = -dist
		}

		candidates = append(candidates, candidate{ticker: m.Ticker, bid: yesBid, dist: dist})
	}

	nCandidates = len(candidates)
	if nCandidates == 0 {
		return "", 0, currentActiveBid, 0, nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	best := candidates[0]
	return best.ticker, best.bid, currentActiveBid, nCandidates, nil
}

// scan is a thin helper used by game.go to pick the initial market before
// starting the WebSocket feed.
func (s *MarketScanner) scan(ctx context.Context) (string, int, error) {
	best, bid, _, _, err := s.scanWithCurrent(ctx, "")
	return best, bid, err
}
