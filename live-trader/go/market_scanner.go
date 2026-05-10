package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"sort"
	"time"
)

// MarketScanner continuously polls the Kalshi REST API for markets tied to an event.
// It finds the most liquid market with a yes_bid closest to 50 cents, and emits it
// to a channel if it changes and the old market is out of bounds.
type MarketScanner struct {
	eventTicker string
	minYesBid   int
	maxYesBid   int
}

func NewMarketScanner(eventTicker string, minYesBid, maxYesBid int) *MarketScanner {
	return &MarketScanner{
		eventTicker: eventTicker,
		minYesBid:   minYesBid,
		maxYesBid:   maxYesBid,
	}
}

// Run polls every 10 seconds and sends new market tickers to outCh.
// It returns immediately if eventTicker is empty.
func (s *MarketScanner) Run(ctx context.Context, initialTicker string, outCh chan<- string) {
	if s.eventTicker == "" {
		return
	}

	currentTicker := initialTicker
	currentBid := 50 // assume middle initially until we get real data

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	// Initial scan right away
	best, bid, err := s.scan(ctx)
	if err == nil && best != "" && best != currentTicker {
		currentTicker = best
		currentBid = bid
		outCh <- best
		log.Printf("[MARKET SWAP] Initial market selected: %s (@ %d¢)", best, bid)
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// 1. Check if we need to swap (is current market out of bounds?)
			if currentBid >= s.minYesBid && currentBid <= s.maxYesBid {
				// We still need to update currentBid for the active ticker so we know when it drifts
				_, _, err := s.scan(ctx)
				if err == nil {
					// We just use the scan to update our mental model of the active market's price.
					// Actually, scan() returns the BEST market. We should instead fetch the specific current ticker.
					// To keep it simple, we just always scan for the best market.
					// If the best market is the one we are already on, great.
					// If it's a different one, we only swap if the current one is out of bounds.
					// Let's refine this: scan() returns the best market and its bid, and also the bid of the current market.
				}
			}

			// Let's just run a fresh scan
			best, bestBid, currentActiveBid, err := s.scanWithCurrent(ctx, currentTicker)
			if err != nil {
				zlog.Warn().Err(err).Msg("market scanner failed to poll")
				continue
			}

			currentBid = currentActiveBid

			// 2. Decide if we swap
			// We swap if the current market is out of bounds AND we found a better one
			if (currentBid < s.minYesBid || currentBid > s.maxYesBid) && best != "" && best != currentTicker {
				log.Printf("[MARKET SWAP] %s drifted to %d¢. Swapping to %s (@ %d¢)", currentTicker, currentBid, best, bestBid)
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

func (s *MarketScanner) scanWithCurrent(ctx context.Context, currentTicker string) (bestTicker string, bestBid int, currentActiveBid int, err error) {
	requestURL := fmt.Sprintf("%s?event_ticker=%s", kalshiRESTURL("/markets"), url.QueryEscape(s.eventTicker))

	req, err := http.NewRequestWithContext(ctx, "GET", requestURL, nil)
	if err != nil {
		return "", 0, 0, err
	}

	headers, err := GetKalshiAuthHeaders("GET", "/markets")
	if err != nil {
		return "", 0, 0, err
	}
	for k, v := range headers {
		req.Header[k] = v
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", 0, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, fmt.Errorf("kalshi API returned %d", resp.StatusCode)
	}

	var data kalshiMarketResp
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", 0, 0, err
	}

	type candidate struct {
		ticker string
		bid    int
		dist   int
	}

	var candidates []candidate
	currentActiveBid = 50 // default

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

		candidates = append(candidates, candidate{
			ticker: m.Ticker,
			bid:    yesBid,
			dist:   dist,
		})
	}

	if len(candidates) == 0 {
		return "", 0, currentActiveBid, nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	best := candidates[0]
	return best.ticker, best.bid, currentActiveBid, nil
}

// scan is a helper for initial startup
func (s *MarketScanner) scan(ctx context.Context) (string, int, error) {
	best, bid, _, err := s.scanWithCurrent(ctx, "")
	return best, bid, err
}
