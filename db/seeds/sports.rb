sports = [
  { code: 'baseball', name: 'Baseball' },
  { code: 'basketball', name: 'Basketball' },
  { code: 'football', name: 'Football' },
  { code: 'hockey', name: 'Hockey' },
  { code: 'soccer', name: 'Soccer' },
  { code: 'tennis', name: 'Tennis' },
  { code: 'golf', name: 'Golf' }
]

sports.each do |attrs|
  Sport.find_or_create_by!(code: attrs[:code]) { |s| s.name = attrs[:name] }
end