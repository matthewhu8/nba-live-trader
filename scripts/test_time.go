package main

import (
	"fmt"
	"strings"
	"time"
)

func main() {
	gameTimeUTC := "2026-05-06T23:30:00Z"
	loc, _ := time.LoadLocation("America/New_York")
	t, _ := time.Parse(time.RFC3339, gameTimeUTC)
	tEST := t.In(loc)
	dateStr := strings.ToUpper(tEST.Format("06Jan02"))
	
	fmt.Printf("KXNBASPREAD-%s%s%s\n", dateStr, "MIN", "SAS")
}
