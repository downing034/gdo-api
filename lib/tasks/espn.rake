# lib/tasks/espn.rake
namespace :espn do
  desc "Fetch games and odds from ESPN for a league and date"
  task :fetch_games, [:league, :date] => :environment do |t, args|
    league = args[:league] || 'ncaam'
    date = args[:date] ? Date.parse(args[:date]) : Date.current

    puts "=" * 60
    puts "Fetching #{league.upcase} games for #{date}..."
    puts "=" * 60

    service = Espn::GamesService.new(league_code: league, date: date)
    results = service.call

    puts "\n#{"=" * 60}"
    puts "Results"
    puts "=" * 60
    puts "Games created:   #{results[:games_created]}"
    puts "Games updated:   #{results[:games_updated]}"
    puts "Games skipped:   #{results[:games_skipped]}"
    puts "Odds created:    #{results[:odds_created]}"
    puts "Odds skipped:    #{results[:odds_skipped]}"
    puts "Results created: #{results[:results_created]}"
    puts "Teams created:   #{results[:teams_created]}"

    if results[:errors].any?
      puts "\nErrors (#{results[:errors].count}):"
      results[:errors].each do |error|
        puts "  Event #{error[:event_id]}: #{error[:error]}"
      end
    end
  end

  desc "Fetch games for a date range"
  task :fetch_games_range, [:league, :start_date, :end_date] => :environment do |t, args|
    league = args[:league] || 'ncaam'
    start_date = Date.parse(args[:start_date])
    end_date = Date.parse(args[:end_date])

    total_results = {
      games_created: 0,
      games_updated: 0,
      odds_created: 0,
      results_created: 0,
      teams_created: 0,
      errors: []
    }

    (start_date..end_date).each do |date|
      puts "\n#{'=' * 60}"
      puts "#{date}"
      puts '=' * 60

      service = Espn::GamesService.new(league_code: league, date: date)
      results = service.call

      total_results[:games_created] += results[:games_created]
      total_results[:games_updated] += results[:games_updated]
      total_results[:odds_created] += results[:odds_created]
      total_results[:results_created] += results[:results_created]
      total_results[:teams_created] += results[:teams_created]
      total_results[:errors].concat(results[:errors])

      puts "Created: #{results[:games_created]}, Odds: #{results[:odds_created]}"

      sleep 1 # Be nice to ESPN
    end

    puts "\n#{'=' * 60}"
    puts "TOTAL RESULTS"
    puts '=' * 60
    puts "Games created:   #{total_results[:games_created]}"
    puts "Games updated:   #{total_results[:games_updated]}"
    puts "Odds created:    #{total_results[:odds_created]}"
    puts "Results created: #{total_results[:results_created]}"
    puts "Teams created:   #{total_results[:teams_created]}"
    puts "Errors:          #{total_results[:errors].count}"
  end
end