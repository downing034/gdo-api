baseball = Sport.find_by!(code: 'baseball')
basketball = Sport.find_by!(code: 'basketball')
football = Sport.find_by!(code: 'football')
hockey = Sport.find_by!(code: 'hockey')

leagues = [
  { sport: baseball, code: 'mlb', name: 'Major League Baseball', display_name: 'MLB', has_conferences: true, active: true },
  { sport: basketball, code: 'ncaam', name: 'NCAA Men\'s Basketball', display_name: 'NCAAM', has_conferences: true, active: true },
  { sport: football, code: 'ncaaf', name: 'NCAA Football', display_name: 'NCAAF', has_conferences: true, active: true },
  { sport: football, code: 'nfl', name: 'National Football League', display_name: 'NFL', has_conferences: true, active: true },
  { sport: basketball, code: 'nba', name: 'National Basketball Association', display_name: 'NBA', has_conferences: true, active: false },
  { sport: hockey, code: 'nhl', name: 'National Hockey League', display_name: 'NHL', has_conferences: true, active: false }
]

leagues.each do |attrs|
  League.find_or_create_by!(code: attrs[:code]) do |l|
    l.name = attrs[:name]
    l.display_name = attrs[:display_name]
    l.has_conferences = attrs[:has_conferences]
    l.active = attrs[:active]
    l.sport = attrs[:sport]
  end
end