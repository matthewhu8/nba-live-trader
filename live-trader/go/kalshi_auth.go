package main

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// GetKalshiAuthHeaders generates the three RSA-PSS signed request headers.
func GetKalshiAuthHeaders(method, path string) (http.Header, error) {
	keyID := os.Getenv("KALSHI_KEY_ID")
	pemPath := os.Getenv("KALSHI_PEM_PATH")

	if keyID == "" {
		return nil, fmt.Errorf("KALSHI_KEY_ID env var not set")
	}
	if pemPath == "" {
		return nil, fmt.Errorf("KALSHI_PEM_PATH env var not set")
	}

	pemBytes, err := os.ReadFile(pemPath)
	if err != nil {
		return nil, fmt.Errorf("read PEM %q: %w", pemPath, err)
	}

	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, fmt.Errorf("no PEM block found in %q", pemPath)
	}

	rsaKey, err := parseRSAKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	tsMS := strconv.FormatInt(time.Now().UnixMilli(), 10)
	message := []byte(tsMS + strings.ToUpper(method) + kalshiSignPath(path))
	digest := sha256.Sum256(message)

	sig, err := rsa.SignPSS(rand.Reader, rsaKey, crypto.SHA256, digest[:], &rsa.PSSOptions{
		SaltLength: rsa.PSSSaltLengthEqualsHash,
	})
	if err != nil {
		return nil, fmt.Errorf("sign PSS: %w", err)
	}

	h := http.Header{}
	h.Set("KALSHI-ACCESS-KEY", keyID)
	h.Set("KALSHI-ACCESS-TIMESTAMP", tsMS)
	h.Set("KALSHI-ACCESS-SIGNATURE", base64.StdEncoding.EncodeToString(sig))
	return h, nil
}

// parseRSAKey tries PKCS8 first, then PKCS1.
func parseRSAKey(derBytes []byte) (*rsa.PrivateKey, error) {
	key, err := x509.ParsePKCS8PrivateKey(derBytes)
	if err == nil {
		rsaKey, ok := key.(*rsa.PrivateKey)
		if !ok {
			return nil, fmt.Errorf("PKCS8 key is not RSA")
		}
		return rsaKey, nil
	}

	rsaKey, pkcs1Err := x509.ParsePKCS1PrivateKey(derBytes)
	if pkcs1Err != nil {
		return nil, fmt.Errorf("parse private key (PKCS8: %v, PKCS1: %v)", err, pkcs1Err)
	}
	return rsaKey, nil
}
