// MarketScanner polls the Kalshi REST API for the markets under one event
// (all the spreads in KXNBASPREAD-26MAY10NYKPHI, say) and picks the one to
// subscribe to: the active market whose yes_bid sits closest to 50¢.
//
// It then pins that market for the rest of the game and only swaps if the bid
// drifts outside [driftLowBid, driftHighBid]. Sibling markets like SAS1 and OKC1
// are logically opposite bets, so hopping between them on bid noise would flip
// what the sign of the model's trajectory means. The MMoE was trained on one
// stable market per game.
//
// Every poll emits a market_scan record and every swap a market_swap record, so
// what the scanner saw at any moment is answerable from the logs alone.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"sync/atomic"
	"time"
)

type MarketScanner struct {
	eventTicker  string
	driftLowBid  int         // swap if the locked market's bid drops below this
	driftHighBid int         // swap if the locked market's bid rises above this
	jsonLog      *JSONLogger // nil-safe
	gameID       string      // for JSONL game_id field
}

func NewMarketScanner(eventTicker string, driftLowBid, driftHighBid int, jsonLog *JSONLogger, gameID string) *MarketScanner {
	// Fall back to a permissive band when the YAML omits these. Tighter risks
	// thrash; looser defeats the point of the drift trigger.
	if driftLowBid <= 0 {
		driftLowBid = 20
	}
	if driftHighBid <= 0 || driftHighBid > 99 {
		driftHighBid = 80
	}
	return &MarketScanner{
		eventTicker:  eventTicker,
		driftLowBid:  driftLowBid,
		driftHighBid: driftHighBid,
		jsonLog:      jsonLog,
		gameID:       gameID,
	}
}

// Run polls every 10 seconds and sends a new market ticker on outCh whenever a
// swap is warranted. Returns immediately if eventTicker is empty.
//
// While positionOpen is set the scanner holds its market even if the bid drifts
// out of band, because exits price off the active book: swapping mid-position
// would compute the exit from a different strike than the one we hold.
func (s *MarketScanner) Run(ctx context.Context, initialTicker string, outCh chan<- string, positionOpen *atomic.Bool) {
	if s.eventTicker == "" {
		return
	}

	currentTicker := initialTicker
	currentBid := 50 // assumed mid until the first poll fills it in

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

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
			// In band means the pinned market has not drifted far enough to abandon.
			// The band is deliberately wider than the entry band: holding the same
			// logical bet at a lopsided price beats swapping to a sibling contract.
			inBand := currentBid >= s.driftLowBid && currentBid <= s.driftHighBid

			// Drift outside the band is the only swap trigger.
			decision := "stay"
			reason := ""
			switch {
			case !inBand && best == "":
				decision = "no_candidates"
			case !inBand && best == currentTicker:
				decision = "best_is_current"
			case !inBand:
				decision = "swap"
				reason = "out_of_band"
			}

			// Defer swaps while a position is open; resume once flat.
			if decision == "swap" && positionOpen != nil && positionOpen.Load() {
				decision = "swap_deferred_open"
			}

			s.emitScan(currentTicker, currentBid, best, bestBid, nCandidates, inBand, decision)

			if decision == "swap" {
				s.emitSwap(currentTicker, best, currentBid, bestBid, reason)
				zlog.Info().
					Str("old_market", currentTicker).
					Int("old_bid_cents", currentBid).
					Str("new_market", best).
					Int("new_bid_cents", bestBid).
					Msg("market scanner: swapping (drift outside band)")
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

// emitScan writes one market_scan line on every poll, swap or not, so the trace
// is continuous.
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

// emitSwap writes one market_swap line when a new ticker is pushed onto outCh.
// Reasons are "initial" and "out_of_band".
func (s *MarketScanner) emitSwap(oldT, newT string, oldBid, newBid int, reason string) {
	if s.jsonLog == nil {
		return
	}
	s.jsonLog.Emit("market_swap", s.gameID, map[string]interface{}{
		"old_ticker": oldT,
		"new_ticker": newT,
		"old_bid":    oldBid,
		"new_bid":    newBid,
		"reason":     reason,
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
//   - bestTicker:       active market with bid closest to 50¢ that clears the
//     liquidity and spread filters, or "" if none qualify
//   - bestBid:          that market's yes_bid in cents
//   - currentActiveBid: live yes_bid of currentTicker, or 50¢ if it wasn't found
//   - nCandidates:      how many markets passed the filters this poll
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
	currentActiveBid = 50 // overwritten if currentTicker is found

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

// scan picks the initial market before the WebSocket feed starts.
func (s *MarketScanner) scan(ctx context.Context) (string, int, error) {
	best, bid, _, _, err := s.scanWithCurrent(ctx, "")
	return best, bid, err
}
