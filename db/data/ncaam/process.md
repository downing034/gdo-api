# NCAAM Data Processing Guide

This guide covers collecting raw data from Barthag (barttorvik.com) and processing it into the final CSV files used for analysis.

---

## Overview

**Input Files (Raw Data):**
| File | Source | Description |
|------|--------|-------------|
| `ncaam_barthag_team_stats_raw.csv` | Team Stats page | Team season stats with ranks (double header format) |
| `ncaam_XX_XX_games_raw.csv` | Game Finder | Game-by-game stats (two rows per game) |
| `ncaam_barthag_teams_table_raw.csv` | Team Tables | Additional team metrics (Barthag, SOS, height, etc.) |

**Output Files:**
| File | Description |
|------|-------------|
| `base_model_game_data_with_rolling.csv` | Merged game data with rolling averages |
| `ncaam-game-data-final.csv` | Team season summary |

---

## Step 1: Collect Team Stats

**Source:** [Team Stats Page](https://barttorvik.com/teamstats.php?year=2026&sort=2)

> ⚠️ Ensure the year parameter matches the current season (e.g., `year=2026` for 2025-26 season)

### Instructions:
1. Navigate to the Team Stats page
2. Select all data in the table (Ctrl+A or Cmd+A)
3. Copy and paste into Google Sheets (or Excel)
4. Export as CSV: `ncaam_barthag_team_stats_raw.csv`

### Format Notes:
- This file has a **double header row** (parent categories + sub-categories)
- Each stat cell contains **value + rank** separated by a newline (e.g., `121.7\n23`)
- Duplicate header rows appear every ~100 teams (for web readability) — the script removes these

### Example Structure:
```
,,,Adj. Eff.,,Eff. FG%,,Turnover%,,...
Rk,Team,Conf,Off.,Def.,Off.,Def.,Off.,Def.,...
1,Colorado St.,MWC,"121.7\n23","111.4\n241",...
```

---

## Step 2: Collect Game-by-Game Stats

**Source:** [Game Finder](https://barttorvik.com/gamestat.php?sIndex=7&year=2026&tvalue=All&cvalue=All&opcvalue=All&ovalue=All&minwin=All&mindate=&maxdate=&typev=All&venvalue=All&minadjo=0&minadjd=200&mintempo=0&minppp=0&minefg=0&mintov=200&minreb=0&minftr=0&minpppd=200&minefgd=200&mintovd=0&minrebd=200&minftrd=200&mings=0&mingscript=-100&maxx=100&coach=All&opcoach=All&adjoSelect=min&adjdSelect=max&tempoSelect=min&pppSelect=min&efgSelect=min&tovSelect=max&rebSelect=min&ftrSelect=min&pppdSelect=max&efgdSelect=max&tovdSelect=min&rebdSelect=max&ftrdSelect=max&gscriptSelect=min&sortToggle=1)

### Instructions:

1. **Adjust filters** to include all games:
   - Set date range as wide as possible
   - Set max display to 4000 (though it caps at 2500 results)

2. **Pagination workaround** (if more than 2500 games):
   - Collect the first 2500 games
   - Note the date of the last game collected
   - Adjust the start date filter to that date and collect the next batch
   - Repeat until all games are captured

3. **Copy and paste into Google Sheets**

4. **Fix the last column header:**
   - Sheets may show `#ERROR!` for the last column
   - Rename it to `+/-` (this is the average lead/deficit)

5. **Export as CSV:** `ncaam_25_26_games_raw.csv`
   - Use the season years in the filename (e.g., `25_26` for 2025-26)

### Format Notes:
- This file has a **double header row** (Offense/Defense categories)
- **Each game has TWO rows** — one from each team's perspective
- All stats in a row are from the "Team" column's perspective

### Key Columns:
| Column | Description |
|--------|-------------|
| Team | The team whose perspective this row represents |
| Opp. | The opponent |
| Venue | H (home), A (away), or N (neutral) — from Team's perspective |
| Result | W/L and score (e.g., "W, 64-56") |
| T | Tempo |
| G-Sc | Game Score (0-100, higher = better performance) |
| +/- | Average lead/deficit during the game |
| Offense cols | Team's offensive stats for this game |
| Defense cols | Team's defensive stats for this game |

### How Games Get Merged:
The script pairs the two rows for each game and creates a single record:
- **Non-neutral games:** Row with `Venue=H` → home team, `Venue=A` → away team
- **Neutral games:** Both rows have `Venue=N`; first row encountered → away, second → home; final `venue` column = "N"

---

## Step 3: Collect Team Table

**Source:** [Team Tables](https://barttorvik.com/team-tables_each.php)

### Instructions:
1. Navigate to the Team Tables page
2. Copy and paste the table into Google Sheets
3. **Add a header for the first column** — it contains the Barthag rank but has no header; name it `id` or similar
4. Export as CSV: `ncaam_barthag_teams_table_raw.csv`

### Key Columns Provided:
| Column | Maps To |
|--------|---------|
| Barthag | Barthag rating |
| Adj OE / Adj DE | Adjusted Offensive/Defensive Efficiency |
| Raw T / Adj. T | Raw and Adjusted Tempo |
| Blk % / Blked % | Block% Def / Block% Off |
| Ast % / Op Ast % | Assist% Off / Assist% Def |
| Elite SOS | Strength of Schedule (joined to games) |
| Avg Hgt. / Eff. Hgt. | Height metrics |
| Exp. | Experience |
| Talent | Talent rating |
| PPP Off. / PPP Def. | Points Per Possession |

---

## Step 4: Run the Processing Script

### File Placement:
Place all three raw CSV files in your data directory.

### Command:
```bash
python ncaam_data_processor.py \
    --games path/to/ncaam_25_26_games_raw.csv \
    --team-stats path/to/ncaam_barthag_team_stats_raw.csv \
    --team-table path/to/ncaam_barthag_teams_table_raw.csv \
    --output-dir path/to/output/
```

### Optional Arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--games-output` | `base_model_game_data_with_rolling.csv` | Output filename for games |
| `--team-output` | `ncaam-game-data-final.csv` | Output filename for team summary |

### Example for 2025-26 Season:
```bash
python ncaam_data_processor.py \
    --games ncaam_25_26_games_raw.csv \
    --team-stats ncaam_barthag_team_stats_raw.csv \
    --team-table ncaam_barthag_teams_table_raw.csv \
    --output-dir ../ncaam
```

---

## Output File Descriptions

### `base_model_game_data_with_rolling.csv`

One row per game with:
- Game info: id, date, away/home teams, conferences, venue, scores
- Team stats for both sides (AdjO, AdjD, Tempo, Four Factors, G-Score)
- **Rolling averages** (5-game and 10-game) for AdjO, AdjD, Tempo
- **Strength of Schedule** (Elite SOS) joined by team name
- **home_advantage** constant (3)

### `ncaam-game-data-final.csv`

One row per team with:
- Team ID, name, conference
- All efficiency metrics with ranks
- Advanced stats (height, experience, talent)
- Barthag, SOS, PPP

---

## Troubleshooting

### Missing Conference or Ranks
If a team appears in the games or team table but not in the team stats file, their conference and rank columns will be blank. Ensure the team stats file is complete.

### Missing SOS Values
If `away_sos` or `home_sos` is blank, the team wasn't found in the team table. Check for team name mismatches between files.

### Date Parsing Issues
Dates should be in `MM/D/YY` or `MM/DD/YY` format. The script converts them to `YYYY-MM-DD`.

---

## Updating for a New Season

1. Update the year parameter in the Barthag URLs
2. Rename the games file with new season years (e.g., `ncaam_26_27_games_raw.csv`)
3. Re-collect all three source files
4. Run the script with the new `--games` filename


# NCAAM Daily Workflow

## Updating Past Games (Yesterday or Earlier)

1. **Fetch game results from ESPN**
   ```bash
   rails espn:fetch_games[ncaam,2026-01-20]
   ```

2. **Export to clipboard (no headers) for spreadsheet**
   ```bash
   rails games:export_ncaam[2026-01-20,false] | pbcopy
   ```
   > ⚠️ Predictions must already be in GDO API or they will be overwritten when pasting into the spreadsheet.

3. **If predictions are missing**, run the backfill job first:
   ```bash
   rails games:backfill_ncaam[2026-01-20]
   ```
   Then repeat step 2.

---

## Adding Today's or Future Games

1. **Fetch games and odds from ESPN**
   ```bash
   rails espn:fetch_games[ncaam,2026-01-21]
   ```
   > ⚠️ Odds availability:
   > - NCAAM: Current day only
   > - NFL/NCAAF: Current week

2. **Export to clipboard for spreadsheet**
   ```bash
   rails games:export_ncaam[2026-01-21,false] | pbcopy
   ```

3. **Add predictions manually** (currently manual process)
   - SL predictions (all leagues)
   - GDO predictions (NCAAM only)

4. **Export CSV and save to data directory**
   - Save as `db/data/ncaam_model_tracking.csv`

5. **Run backfill to add predictions to database**
   ```bash
   rails games:backfill_ncaam[2026-01-21]
   ```

---

## Get Sweet Spot Picks

```bash
# Default (today's picks, yesterday's results, all-time stats)
rails ncaam:sweet_spot

# Custom dates
rails ncaam:sweet_spot[2026-01-22,2026-01-21,2026-01-10,2026-01-21]
```

**Arguments (all optional):**
1. `today_date` - date for today's picks (default: today)
2. `yesterday_date` - date for yesterday's results (default: yesterday)
3. `stats_start` - start date for accuracy stats (default: 2026-01-02)
4. `stats_end` - end date for accuracy stats (default: yesterday)

---

## Quick Reference

| Task | Command |
|------|---------|
| Fetch ESPN games | `rails espn:fetch_games[ncaam,DATE]` |
| Export to clipboard | `rails games:export_ncaam[DATE,false] \| pbcopy` |
| Export with headers | `rails games:export_ncaam[DATE,true] \| pbcopy` |
| Backfill predictions | `rails games:backfill_ncaam[DATE]` |
| Sweet spot analysis | `rails ncaam:sweet_spot` |