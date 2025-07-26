mlb = League.find_by!(code: 'mlb')

mlb_teams = [
  { code: 'ATH', location_name: '',  nickname: 'Athletics' },
  { code: 'ARI', location_name: 'Arizona', nickname: 'Diamondbacks' },
  { code: 'ATL', location_name: 'Atlanta', nickname: 'Braves' },
  { code: 'BAL', location_name: 'Baltimore', nickname: 'Orioles' },
  { code: 'BOS', location_name: 'Boston', nickname: 'Red Sox' },
  { code: 'CHC', location_name: 'Chicago', nickname: 'Cubs' },
  { code: 'CHW', location_name: 'Chicago', nickname: 'White Sox' },
  { code: 'CIN', location_name: 'Cincinnati', nickname: 'Reds' },
  { code: 'CLE', location_name: 'Cleveland', nickname: 'Guardians' },
  { code: 'COL', location_name: 'Colorado', nickname: 'Rockies' },
  { code: 'DET', location_name: 'Detroit', nickname: 'Tigers' },
  { code: 'HOU', location_name: 'Houston', nickname: 'Astros' },
  { code: 'KCR', location_name: 'Kansas City', nickname: 'Royals' },
  { code: 'LAA', location_name: 'Los Angeles', nickname: 'Angels' },
  { code: 'LAD', location_name: 'Los Angeles', nickname: 'Dodgers' },
  { code: 'MIA', location_name: 'Miami', nickname: 'Marlins' },
  { code: 'MIL', location_name: 'Milwaukee', nickname: 'Brewers' },
  { code: 'MIN', location_name: 'Minnesota', nickname: 'Twins' },
  { code: 'NYM', location_name: 'New York', nickname: 'Mets' },
  { code: 'NYY', location_name: 'New York', nickname: 'Yankees' },
  { code: 'PHI', location_name: 'Philadelphia', nickname: 'Phillies' },
  { code: 'PIT', location_name: 'Pittsburgh', nickname: 'Pirates' },
  { code: 'SEA', location_name: 'Seattle', nickname: 'Mariners' },
  { code: 'SDP', location_name: 'San Diego', nickname: 'Padres' },
  { code: 'SFG', location_name: 'San Francisco', nickname: 'Giants' },
  { code: 'STL', location_name: 'Saint Louis', nickname: 'Cardinals' },
  { code: 'TBR', location_name: 'Tampa Bay', nickname: 'Rays' },
  { code: 'TEX', location_name: 'Texas', nickname: 'Rangers' },
  { code: 'TOR', location_name: 'Toronto', nickname: 'Blue Jays' },
  { code: 'WSN', location_name: 'Washington', nickname: 'Nationals' }
]

mlb_teams.each do |attrs|
  Team.find_or_create_by!(code: attrs[:code], league: mlb) do |t|
    t.location_name = attrs[:location_name]
    t.nickname = attrs[:nickname]
    t.active = true
  end
end