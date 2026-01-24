# ESPN Games Fetch
rails espn:fetch_games[ncaam]                              # Today's NCAAM games
rails espn:fetch_games[ncaam,2026-01-23]                   # Specific date
rails espn:fetch_games[nba,2026-01-23]                     # NBA specific date
rails espn:fetch_games_range[ncaam,2026-01-20,2026-01-25]  # Date range

# NCAAM Data & Model
rails ncaam:refresh                                        # Process CSVs + train model
rails ncaam:predict                                        # Generate predictions for upcoming games

# Sweet Spot Analysis
rails ncaam:sweet_spot                                     # Defaults (today/yesterday)
rails ncaam:sweet_spot[2026-01-23,2026-01-22,2026-01-01,2026-01-22]  # [today, yesterday, stats_start, stats_end]

# Export Games to CSV
rails games:export_ncaam                                   # Today
rails games:export_ncaam[2026-01-23]                       # Specific date
rails games:export_ncaam[2026-01-23,true]                  # With header

# Backfill from CSV
rails games:backfill_ncaam                                 # Yesterday to today
rails games:backfill_ncaam[2026-01-20,2026-01-22]          # Date range
rails games:backfill_mlb
rails games:backfill_ncaaf
rails games:backfill_nfl