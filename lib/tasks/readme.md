# ESPN Games Fetch
rails espn:fetch_games[ncaam]                              # Today's NCAAM games
rails espn:fetch_games[ncaam,2026-01-23]                   # Specific date
rails espn:fetch_games[nba,2026-01-23]                     # NBA specific date
rails espn:fetch_games_range[ncaam,2026-01-20,2026-01-25]  # Date range

# NCAAM Data & Model
rake ncaam:process                                         # Process CSVs (current season only)
rake ncaam:process SEASONS=24_25,25_26                     # Process multiple seasons for training
rake ncaam:process SEASONS=23_24,24_25,25_26               # Process three seasons when ready
rake ncaam:train                                           # Train model
rake ncaam:refresh                                         # Process CSVs + train model
rake ncaam:predict                                         # Generate predictions for upcoming games

# Sweet Spot Analysis
rails ncaam:sweet_spot                                     # Defaults (today/yesterday)
rails ncaam:sweet_spot[2026-01-23,2026-01-22,2026-01-01,2026-01-22]  # [today, yesterday, stats_start, stats_end]

# Export Games to CSV
rails games:export_ncaam                                   # Today
rails games:export_ncaam[2026-01-23]                       # Specific date
rails games:export_ncaam[2026-01-23,true]                  # With header
rails games:export_ncaam[2026-01-23] | pbcopy              # directly to clipboard

# Export Stale Games
rails games:export_stale[ncaam]                            # Export stale NCAAM games to CSV
rails games:export_stale[nfl]                              # Export stale NFL games to CSV

# Backfill from CSV
rails games:backfill_ncaam                                 # Yesterday to today
rails games:backfill_ncaam[2026-01-20,2026-01-22]          # Date range
rails games:backfill_mlb
rails games:backfill_ncaaf
rails games:backfill_nfl