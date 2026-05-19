import json

# Simulated Kalshi API response for an event with multiple spread lines
mock_markets = [
    {"ticker": "SAS10", "yes_bid": 85, "yes_ask": 88, "volume": 1500},
    {"ticker": "SAS13", "yes_bid": 72, "yes_ask": 75, "volume": 2000},
    {"ticker": "SAS16", "yes_bid": 45, "yes_ask": 48, "volume": 5000},  # Close to 50
    {"ticker": "SAS19", "yes_bid": 25, "yes_ask": 28, "volume": 800},
    {"ticker": "SAS22", "yes_bid": 10, "yes_ask": 12, "volume": 100},
    {"ticker": "SAS25", "yes_bid": 49, "yes_ask": 51, "volume": 0},     # Perfect price, but 0 volume (illiquid)
    {"ticker": "SAS28", "yes_bid": 50, "yes_ask": 90, "volume": 10},    # Perfect bid, but 40 cent spread (illiquid)
]

def scan_markets(current_ticker, current_bid):
    print(f"--- SCAN INITIATED ---")
    print(f"Current Market: {current_ticker} @ {current_bid}¢")
    
    if 30 <= current_bid <= 70:
        print("Result: Current market is within 30¢-70¢ threshold. NO SWAP NEEDED.")
        return current_ticker
        
    print(f"Result: Current market ({current_bid}¢) is outside threshold! Searching for replacement...\n")
    
    valid_markets = []
    print("Filtering markets:")
    for m in mock_markets:
        bid = m["yes_bid"]
        ask = m["yes_ask"]
        vol = m["volume"]
        spread = ask - bid
        
        if vol == 0:
            print(f"  [REJECTED] {m['ticker']} (Volume is 0)")
            continue
        if spread > 10:
            print(f"  [REJECTED] {m['ticker']} (Spread is too wide: {spread}¢)")
            continue
            
        distance = abs(bid - 50)
        valid_markets.append((distance, m))
        print(f"  [ACCEPTED] {m['ticker']} (Bid: {bid}¢, Spread: {spread}¢, Vol: {vol}) -> Distance from 50¢: {distance}¢")
        
    if not valid_markets:
        print("\nResult: No liquid replacement found. HOLDING current market.")
        return current_ticker
        
    # Sort by distance from 50 cents
    valid_markets.sort(key=lambda x: x[0])
    best_market = valid_markets[0][1]
    
    print(f"\nResult: Best alternative found! Issuing WebSocket swap command.")
    print(f"  -> Sending: {{\"cmd\": \"update_subscription\", \"action\": \"delete_markets\", \"markets\": [\"{current_ticker}\"]}}")
    print(f"  -> Sending: {{\"cmd\": \"update_subscription\", \"action\": \"add_markets\", \"markets\": [\"{best_market['ticker']}\"]}}")
    
    return best_market['ticker']

print("SCENARIO 1: Game is stable.")
scan_markets("SAS16", 55)

print("\n\nSCENARIO 2: Spurs go on a massive run. SAS16 price spikes to 80 cents!")
scan_markets("SAS16", 80)
