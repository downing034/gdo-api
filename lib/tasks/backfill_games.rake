# lib/tasks/backfill_games.rake
require 'csv'

namespace :games do
  # Current week constants - update these as seasons progress
  MAX_NFL_WEEK = 16
  MAX_NCAAF_WEEK = 16

  desc "Backfill MLB games from CSV file"
  task backfill_mlb: :environment do
    file_path = 'db/data/mlb_model_tracking.csv'
    
    unless File.exist?(file_path)
      puts "Error: CSV file not found at #{file_path}"
      exit 1
    end
    
    mlb = League.find_by!(code: 'mlb')
    puts "Processing MLB games from #{file_path}..."
    
    CSV.foreach(file_path, headers: true) do |row|
      begin
        process_mlb_game_row(row, mlb)
      rescue => e
        puts "\nError processing row #{row['id']}: #{e.message}"
        puts "Row data: #{row.to_h}"
        raise e if Rails.env.development?
      end
    end
    
    puts "\n\nMLB backfill complete!"
    puts "Total games in database: #{Game.count}"
  end

  desc "Backfill NFL games from CSV file"
  task backfill_nfl: :environment do
    backfill_league_games('nfl', MAX_NFL_WEEK, 'db/data/nfl_model_tracking.csv')
  end

  desc "Backfill NCAAF games from CSV file"  
  task backfill_ncaaf: :environment do
    backfill_league_games('ncaaf', MAX_NCAAF_WEEK, 'db/data/ncaaf_model_tracking.csv')
  end

  private

  def process_mlb_game_row(row, league)
    # Parse date and teams
    game_date = Date.strptime(row['date'], '%Y%m%d')
    home_team_code = row['home_team'].upcase
    away_team_code = row['away_team'].upcase
    
    # Skip future games without moneyline data
    if game_date > Date.current && row['moneyline_favorite_team'].blank?
      puts "Skipping future game without moneyline: #{away_team_code} @ #{home_team_code} on #{game_date}"
      return
    end
    
    # Find teams
    home_team = Team.find_by!(code: home_team_code, league: league)
    away_team = Team.find_by!(code: away_team_code, league: league)
    
    # Parse start time or default to 1am of game date until time can be set
    start_time = if row['start_time'].present?
      Time.zone.parse(row['start_time']).utc
    else
      game_date.beginning_of_day.utc + 1.hour  # 1am UTC
    end
    
    # Determine initial status based on whether we have results
    has_results = row['away_result'].present? && row['home_result'].present?
    initial_status = has_results ? 2 : 0  # final : scheduled (using integers since enum commented out)

    season = Season.find_by!(
      league: league,
      start_date: ..game_date,
      end_date: game_date..
    )
    
    # Find or create game
    game = Game.find_or_create_by(
      league: league,
      game_date: game_date,
      home_team: home_team,
      away_team: away_team
    ) do |g|
      g.season = season
      g.start_time = start_time
      g.status = initial_status
      status_text = has_results ? "final" : "scheduled"
      puts "Created: #{away_team_code} @ #{home_team_code} on #{game_date} (#{status_text})"
    end
    
    # Update start_time if we have better data and current time is 1am (placeholder)
    if !game.new_record? && row['start_time'].present? && game.start_time&.hour == 1
      game.update!(start_time: start_time)
      puts "Updated time: #{away_team_code} @ #{home_team_code}"
    end
    
    # Create game result if final scores are present
    if row['away_result'].present? && row['home_result'].present?
      away_score = row['away_result'].to_i
      home_score = row['home_result'].to_i
      
      result = GameResult.find_or_create_by(game: game) do |r|
        r.away_score = away_score
        r.home_score = home_score
        r.final = true
        puts "  Result: #{away_team_code} #{away_score} - #{home_team_code} #{home_score}"
      end
      
      # Update game status to final if we just added results to a scheduled game
      game.update!(status: 2) if game.status == 0 && has_results
    end
  end
 
  def backfill_league_games(league_code, max_week, file_path)
    unless File.exist?(file_path)
      puts "Error: CSV file not found at #{file_path}"
      exit 1
    end
    
    league = League.find_by!(code: league_code)
    puts "Processing #{league_code.upcase} games from #{file_path}..."
    puts "Including games up to and including week #{max_week}"
    
    games_created = 0
    games_updated = 0
    results_created = 0
    games_skipped = 0
    
    CSV.foreach(file_path, headers: true) do |row|
      begin
        result = process_week_game_row(row, league, max_week, games_created, games_updated, results_created, games_skipped)
        games_created, games_updated, results_created, games_skipped = result
      rescue => e
        puts "\nError processing row #{row['id']}: #{e.message}"
        puts "Row data: #{row.to_h}"
        raise e if Rails.env.development?
      end
    end
    
    puts "\n\n#{league_code.upcase} backfill complete!"
    puts "Games created: #{games_created}"
    puts "Games updated: #{games_updated}"
    puts "Results created: #{results_created}"
    puts "Games skipped: #{games_skipped}"
    puts "Total games in database: #{Game.count}"
  end

  TEAM_CODE_EXCEPTIONS = {
    'Linden' => 'Linden',
  }

  def process_week_game_row(row, league, max_week, games_created, games_updated, results_created, games_skipped)
    # Parse basic game info
    week = row['week'].to_i

    # Skip games beyond the current week
    if week > max_week
      puts "Skipping Game: Week #{week} beyond week #{max_week}"
      games_skipped += 1
      return [games_created, games_updated, results_created, games_skipped]
    end

    game_date = Date.strptime(row['date'], '%Y%m%d')
    home_team_code = TEAM_CODE_EXCEPTIONS[row['home_team']] || row['home_team'].upcase
    away_team_code = TEAM_CODE_EXCEPTIONS[row['away_team']] || row['away_team'].upcase
    
    # Skip future games without moneyline data
    if (game_date > Date.current && row['moneyline_favorite_team'].blank?)
      puts "Skipping: Week #{week} #{away_team_code} @ #{home_team_code} future without moneyline"
      games_skipped += 1
      return [games_created, games_updated, results_created, games_skipped]
    end
    
    # Find teams
    home_team = Team.find_by!(code: home_team_code, league: league)
    away_team = Team.find_by!(code: away_team_code, league: league)
    
    # Parse start time (MST format like "17:30")
    start_time = if row['start_time'].present?
      time_parts = row['start_time'].split(':')
      hour = time_parts[0].to_i
      minute = time_parts[1].to_i
      game_date.in_time_zone('America/Denver').change(hour: hour, minute: minute).utc
    else
      game_date.beginning_of_day.utc + 1.hour  # 1am UTC default
    end

    season = Season.find_by!(
      league: league,
      start_date: ..game_date,
      end_date: game_date..
    )
    
    # Find or create game - set status based on whether we have results
    has_results = row['away_result'].present? && row['home_result'].present?
    initial_status = has_results ? 2 : 0  # final : scheduled (using integers since enum commented out)
    
    game = Game.find_or_create_by(
      league: league,
      game_date: game_date,
      home_team: home_team,
      away_team: away_team
    ) do |g|
      g.season = season
      g.start_time = start_time
      g.status = initial_status
      games_created += 1
      status_text = has_results ? "final" : "scheduled"
      puts "Created: Week #{week} #{away_team_code} @ #{home_team_code} on #{game_date} (#{status_text})"
    end
    
    # Update start_time if we have better data and current time is 1am (placeholder)
    if !game.new_record? && row['start_time'].present? && game.start_time&.hour == 1
      game.update!(start_time: start_time)
      games_updated += 1
      puts "Updated time: Week #{week} #{away_team_code} @ #{home_team_code}"
    end
    
    # Create game result if final scores are present
    if row['away_result'].present? && row['home_result'].present?
      away_score = row['away_result'].to_i
      home_score = row['home_result'].to_i
      
      result = GameResult.find_or_create_by(game: game) do |r|
        r.away_score = away_score
        r.home_score = home_score
        r.final = true
        results_created += 1
        puts "  Result: #{away_team_code} #{away_score} - #{home_team_code} #{home_score}"
      end
      
      # Update game status to final if we just added results
      game.update!(status: 2) if game.status == 0 && has_results
    end
    
    [games_created, games_updated, results_created, games_skipped]
  end
end