package main

import (
	"os"
	"strings"
)

const (
	kalshiTradeAPIPrefix      = "/trade-api/v2"
	kalshiWebSocketSignPath   = "/trade-api/ws/v2"
	defaultKalshiRESTBaseURL  = "https://external-api.kalshi.com/trade-api/v2"
	defaultKalshiWebSocketURL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
)

func kalshiRESTBaseURL() string {
	if baseURL := strings.TrimRight(os.Getenv("KALSHI_REST_BASE_URL"), "/"); baseURL != "" {
		return baseURL
	}
	return defaultKalshiRESTBaseURL
}

func kalshiRESTURL(endpoint string) string {
	baseURL := kalshiRESTBaseURL()
	cleanEndpoint := "/" + strings.TrimLeft(endpoint, "/")

	if strings.HasSuffix(baseURL, kalshiTradeAPIPrefix) {
		return baseURL + cleanEndpoint
	}
	return baseURL + kalshiTradeAPIPrefix + cleanEndpoint
}

func kalshiWebSocketURL() string {
	if wsURL := strings.TrimSpace(os.Getenv("KALSHI_WS_URL")); wsURL != "" {
		return wsURL
	}
	return defaultKalshiWebSocketURL
}

func kalshiSignPath(path string) string {
	cleanPath := "/" + strings.TrimLeft(path, "/")
	if queryIndex := strings.Index(cleanPath, "?"); queryIndex >= 0 {
		cleanPath = cleanPath[:queryIndex]
	}
	if strings.HasPrefix(cleanPath, "/trade-api/") {
		return cleanPath
	}
	return kalshiTradeAPIPrefix + cleanPath
}
