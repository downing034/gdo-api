# ESPN Games Fetch
rails espn:fetch_games[ncaam]                                        # Today's NCAAM games
rails espn:fetch_games[ncaam,2026-01-23]                             # Specific date
rails espn:fetch_games[nba,2026-01-23]                               # NBA specific date
rails espn:fetch_games_range[ncaam,2026-01-20,2026-01-25]            # Date range

# NCAAM Data & Model
rake ncaam:process                                                   # Process CSVs (current season only)
rake ncaam:process SEASONS=24_25,25_26                               # Process multiple seasons for training
rake ncaam:process SEASONS=23_24,24_25,25_26                         # Process three seasons when ready
rake ncaam:train                                                     # Train model
rake ncaam:refresh                                                   # Process CSVs + train model
rails ncaam:predict                                                  # Today and future (default)
rails ncaam:predict[2026-01-27]                                      # Single date
rails ncaam:predict[2026-01-25,2026-01-27]                           # Date range
rails ncaam:predict[2026-01-25,2026-01-27,true]                      # Date range, include completed
rails ncaam:predict[2026-01-25,,true]                                # Single date, include completed

# Sweet Spot Analysis
rails ncaam:sweet_spot                                               # Defaults (today/yesterday)
rails ncaam:sweet_spot[2026-01-23,2026-01-22,2026-01-01,2026-01-22]  # [today, yesterday, stats_start, stats_end]

# Model team and player data
rails  export:team_stats[ncaam,2026-01-13,2026-01-24]                # Team stats - specific date range
rails  export:team_stats[ncaam,2026-01-24]                           # Team stats - single date (uses as end date, start = end)
rails  export:team_stats[ncaam]                                      # Team stats - defaults (yesterday only)
rails  export:player_stats[ncaam,2026-01-13,2026-01-24]              # Player stats - same patterns

# Export Games to CSV
rails games:export_ncaam                                             # Today
rails games:export_ncaam[2026-01-23]                                 # Specific date
rails games:export_ncaam[2026-01-23,true]                            # With header
rails games:export_ncaam[2026-01-23] | pbcopy                        # directly to clipboard

# Export Stale Games
rails games:export_stale[ncaam]                                      # Export stale NCAAM games to CSV
rails games:export_stale[nfl]                                        # Export stale NFL games to CSV

# Backfill from CSV
rails games:backfill_ncaam                                           # Yesterday to today
rails games:backfill_ncaam[2026-01-20,2026-01-22]                    # Date range
rails games:backfill_mlb
rails games:backfill_ncaaf
rails games:backfill_nfl

# Analyze Model Performance
rails models:performance_report                                      # ncaam, yesterday
rails models:performance_report[ncaam]                               # ncaam, yesterday
rails models:performance_report[ncaam,2026-01-24]                    # ncaam, single date
rails models:performance_report[ncaam,2026-01-20,2026-01-24]         # ncaam, date range
rails models:performance_report[nba,2026-01-24]                      # nba, single date