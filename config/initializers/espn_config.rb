ESPN_LEAGUE_CONFIG = {
  'ncaam' => {
    sport: 'basketball',
    league_path: 'mens-college-basketball',
    groups: 50,
    limit: 300
  },
  'nba' => {
    sport: 'basketball',
    league_path: 'nba',
    groups: 40,
    limit: 100
  },
  'nfl' => {
    sport: 'football',
    league_path: 'nfl',
    groups: nil,  # NFL might not need groups
    limit: 100
  },
  'ncaaf' => {
    sport: 'football',
    league_path: 'college-football',
    groups: 80,
    limit: 200
  },
  'mlb' => {
    sport: 'baseball',
    league_path: 'mlb',
    groups: nil,
    limit: 100
  }
}.freeze