"""
Star player map for the 2025-26 NBA season.

Single source of truth imported by context_features and lineup_features.

All player_ids verified against data/raw/foul_events_202526.parquet.
Advanced stats (BPM, VORP, USG%) sourced from basketball-reference.com/leagues/NBA_2026_advanced.html.
Rankings sourced from HoopsHype preseason top-100 and 2026 NBA All-Star official selections.

2026 All-Star format: USA vs. World. Both pools counted equally as "All-Star selected."
  East starters:  Cunningham, Brunson, Maxey, Giannis, J.Brown
  West starters:  SGA, Curry, Luka, Wembanyama, Jokić
  East reserves:  Mitchell, KAT, Siakam, Duren, J.Johnson, S.Barnes, N.Powell
  West reserves:  Durant, Edwards, Booker, Holmgren, LeBron, Avdija, J.Murray
  Commissioner:   Kawhi Leonard

Tier 1 — franchise stars. Their absence from a lineup materially changes game outcomes.
         All are 2026 All-Star selections OR carry USG% > 30% as primary option.
         33 BPM ≥ 3.5 or VORP ≥ 3.0 for active players; historical franchise status for injured.

Tier 2 — key rotation stars. Starting-caliber impact; their foul trouble shifts game
         dynamics but less dramatically than Tier 1.
         Criteria: All-Star selection OR USG% > 23% as team's primary option with positive BPM.
"""

# ---------------------------------------------------------------------------
# Tier 1 — franchise stars (20 players)
# ---------------------------------------------------------------------------

TIER_1: dict[int, str] = {
    # 2026 All-Star starters — highest impact players in the league
    1628983: "Shai Gilgeous-Alexander",  # OKC — All-Star starter; 33.6% USG, 11.8 BPM, 6.0 VORP (#1 in NBA)
    203999:  "Nikola Jokić",             # DEN — All-Star starter; 6.0 VORP, 8-time All-Star
    1629029: "Luka Dončić",              # LAL — All-Star starter; 37.4% USG, 8.7 BPM, 4.9 VORP
    203507:  "Giannis Antetokounmpo",    # MIL — All-Star starter; 4x MVP candidate
    1641705: "Victor Wembanyama",        # SAS — All-Star starter; generational defender + scorer
    1630595: "Cade Cunningham",          # DET — All-Star starter; 30.9% USG, 6.1 BPM, 3.9 VORP
    1630178: "Tyrese Maxey",             # PHI — All-Star starter; 29.9% USG, 5.9 BPM, 4.5 VORP
    1628973: "Jalen Brunson",            # NYK — All-Star starter; 30.3% USG, franchise PG
    1627759: "Jaylen Brown",             # BOS — All-Star starter; 36.4% USG, 3.5 BPM
    1628378: "Donovan Mitchell",         # CLE — All-Star starter; 32.9% USG, 5.4 BPM, 3.4 VORP

    # 2026 All-Star reserves — elite stars
    201939:  "Stephen Curry",            # GSW — All-Star reserve; all-time 3PT leader, 4x champion
    1630162: "Anthony Edwards",          # MIN — All-Star reserve; 31.5% USG, 4.9 BPM, 3.3 VORP
    1627750: "Jamal Murray",             # DEN — All-Star reserve; 28.6% USG, 4.3 BPM, 3.3 VORP
    1626164: "Devin Booker",             # PHX — All-Star reserve; 32+ USG% scorer, 5x All-Star
    201142:  "Kevin Durant",             # HOU — All-Star reserve (16th selection); 3.9 BPM
    2544:    "LeBron James",             # LAL — All-Star reserve (22nd selection); all-time scorer
    1627783: "Pascal Siakam",            # IND — All-Star reserve; 29.5% USG, 1.5 VORP
    202695:  "Kawhi Leonard",            # LAC — Commissioner's All-Star pick; 33.6% USG when healthy

    # Franchise stars — limited by injury in 2025-26 but lineup-altering when healthy
    203954:  "Joel Embiid",              # PHI — 2x MVP runner-up; 30%+ USG when healthy
    203076:  "Anthony Davis",            # WAS — traded from DAL mid-season; elite two-way big, 27%+ USG
    1628369: "Jayson Tatum",             # BOS — franchise star; returning from Achilles ~Mar 2026; lineup-altering when healthy
}

# ---------------------------------------------------------------------------
# Tier 2 — key rotation stars (33 players)
# ---------------------------------------------------------------------------

TIER_2: dict[int, str] = {
    # 2026 All-Star selections (East reserves)
    1630552: "Jalen Johnson",            # ATL — All-Star reserve; 27.2% USG, 4.1 BPM, 3.0 VORP
    1631105: "Jalen Duren",              # DET — All-Star reserve; elite rim protector
    1626157: "Karl-Anthony Towns",       # NYK — All-Star reserve; 25.5% USG, 2.4 BPM
    1630567: "Scottie Barnes",           # TOR — All-Star reserve; 23.9% USG, 3.6 BPM, 2.9 VORP
    1626181: "Norman Powell",            # MIA — All-Star reserve; key scorer, high efficiency
    1630166: "Deni Avdija",              # POR — All-Star reserve (World pool); 3.1 BPM, versatile wing

    # 2026 All-Star selections (West reserves)
    1631096: "Chet Holmgren",            # OKC — All-Star reserve; elite rim protector + shooter

    # High-USG primary options — lineup anchors whose foul trouble signals market moves
    201935:  "James Harden",             # CLE — traded from LAC (Garland swap); 31.3% USG, 3.8 BPM; primary playmaker
    1630578: "Alperen Şengün",           # HOU — 27.2% USG, 3.9 BPM, 2.7 VORP; dominant center
    1629008: "Michael Porter Jr.",       # BKN — 30.3% USG; primary scorer
    1641718: "Keyonte George",           # UTA — 28.0% USG; franchise primary option
    1631094: "Paolo Banchero",           # ORL — 27.4% USG; franchise star
    1627742: "Brandon Ingram",           # TOR — 27.4% USG; veteran primary scorer
    202331:  "Paul George",              # PHI — veteran anchor; rotation impact

    # Verified from foul_events — strong advanced metrics, lineup-altering defenders/scorers
    1628969: "Mikal Bridges",            # NYK — 3.5 BPM, 3.0 VORP; elite perimeter defender
    1628368: "De'Aaron Fox",             # SAS — 24.9% USG, 2.3 BPM; franchise PG
    1628389: "Bam Adebayo",              # MIA — 24.4% USG, 1.8 BPM; defensive anchor, rim protection
    1630596: "Evan Mobley",              # CLE — DPOY candidate; CLE's defensive cornerstone
    1628384: "OG Anunoby",              # NYK — elite perimeter defender; NYK's defensive backbone
    203944:  "Julius Randle",            # MIN — 26.2% USG, 2.1 VORP; key frontcourt anchor
    203497:  "Rudy Gobert",              # MIN — 3x DPOY; elite rim protector, lineup-defining
    1628374: "Lauri Markkanen",          # UTA — 24.2% USG; franchise star
    1642843: "Cooper Flagg",             # DAL — top 2025 rookie; 25.6% USG, 1.3 VORP
    1630217: "Desmond Bane",             # ORL — 23.3% USG; key scorer
    1630193: "Immanuel Quickley",        # TOR — 21.6% USG, 2.5 BPM, 2.2 VORP; rising star
    1630224: "Jalen Green",              # PHX — primary scorer (traded from HOU)
    1629627: "Zion Williamson",          # NOP — dominant when healthy; 30%+ USG
    1629630: "Ja Morant",               # MEM — franchise PG; top-15 when healthy
    1629636: "Darius Garland",           # LAC — traded from CLE; All-Star caliber PG
    203897:  "Zach LaVine",             # SAC — 30%+ USG scorer, traded from CHI
    1627734: "Domantas Sabonis",         # SAC — triple-double machine; lineup anchor
    1629014: "Anfernee Simons",          # CHI — primary scorer (via POR→BOS→CHI)
    1629632: "Coby White",              # CHI — primary scorer; HoopsHype preseason top-65
    1629027: "Trae Young",              # WAS — traded from ATL; 4x All-Star, 28%+ USG; primary playmaker
    1630532: "Franz Wagner",            # ORL — primary star; All-Star trajectory; 925 poss, 1587 pts in data
    1628991: "Jaren Jackson Jr.",       # UTA — traded from MEM; DPOY candidate; elite rim protector
}

# Combined lookup: player_id → tier (1 or 2)
STAR_PLAYERS: dict[int, int] = {pid: 1 for pid in TIER_1} | {pid: 2 for pid in TIER_2}
