package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

// LoadEnv simply reads a .env file and sets the variables in the current process.
// We do this manually to avoid adding external dependencies like godotenv.
func LoadEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// Skip comments and empty lines
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Split on the first '='
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])

		// Remove quotes if present
		val = strings.Trim(val, `"'`)

		if err := os.Setenv(key, val); err != nil {
			log.Printf("[ENV] Failed to set %s: %v", key, err)
		}
	}

	return scanner.Err()
}

func LoadEnvCandidates(paths ...string) (string, error) {
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			continue
		}
		if err := LoadEnv(path); err != nil {
			return "", err
		}
		return path, nil
	}
	return "", fmt.Errorf("no .env file found in candidates: %s", strings.Join(paths, ", "))
}
