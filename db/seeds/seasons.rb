# Find leagues
nfl = League.find_by!(code: 'nfl')
ncaaf = League.find_by!(code: 'ncaaf')
ncaam = League.find_by!(code: 'ncaam')
mlb = League.find_by!(code: 'mlb')

# Create seasons
Season.find_or_create_by!(league: nfl, name: '2025-26') do |s|
  s.start_date = Date.new(2025, 3, 12)
  s.end_date = Date.new(2026, 3, 11)
  s.active = true
end

Season.find_or_create_by!(league: ncaaf, name: '2025-26') do |s|
  s.start_date = Date.new(2025, 7, 1)
  s.end_date = Date.new(2026, 3, 31)
  s.active = true
end

Season.find_or_create_by!(league: ncaam, name: '2025-26') do |s|
  s.start_date = Date.new(2025, 11, 1)
  s.end_date = Date.new(2026, 4, 30)
  s.active = true
end

Season.find_or_create_by!(league: mlb, name: '2025') do |s|
  s.start_date = Date.new(2025, 2, 1)
  s.end_date = Date.new(2025, 11, 1)
  s.active = false
end

puts "Seasons seeded successfully!"