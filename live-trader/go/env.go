package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

// LoadEnv reads a .env file into the process environment. Done by hand to avoid a
// dependency on godotenv.
func LoadEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
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
