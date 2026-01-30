require 'csv'

namespace :games do
  desc "Backfill MLB games from CSV file"
  task backfill_mlb: :environment do
    file_path = 'db/data/mlb_model_tracking.csv'
    
    unless File.exist?(file_path)
      puts "Error: CSV file not found at #{file_path}"
      exit 1
    end
    
    mlb = League.find_by!(code: 'mlb')
    puts "Processing MLB games from #{file_path}..."
    
    games_created = 0
    games_updated = 0
    results_created = 0
    predictions_created = 0
    
    CSV.foreach(file_path, headers: true) do |row|
      begin
        result = process_mlb_game_row(row, mlb, games_created, games_updated, results_created, predictions_created)
        games_created, games_updated, results_created, predictions_created = result
      rescue => e
        puts "\nError processing row #{row['id']}: #{e.message}"
        puts "Row data: #{row.to_h}"
        raise e if Rails.env.development?
      end
    end
    
    puts "\n\nMLB backfill complete!"
    puts "Games created: #{games_created}"
    puts "Games updated: #{games_updated}"
    puts "Results created: #{results_created}"
    puts "Predictions created: #{predictions_created}"
    puts "Total games in database: #{Game.count}"
  end

  desc "Backfill NFL games from CSV file (default: all games, override with dates)"
  task :backfill_nfl, [:start_date, :end_date] => :environment do |t, args|
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : Date.new(2000, 1, 1)
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : Date.new(2099, 12, 31)
    backfill_date_based_league('nfl', start_date, end_date, 'db/data/nfl_model_tracking.csv')
  end

  desc "Backfill NCAAF games from CSV file (default: all games, override with dates)"
  task :backfill_ncaaf, [:start_date, :end_date] => :environment do |t, args|
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : Date.new(2000, 1, 1)
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : Date.new(2099, 12, 31)
    backfill_date_based_league('ncaaf', start_date, end_date, 'db/data/ncaaf_model_tracking.csv')
  end

  # Task definition
  desc "Backfill NCAAM games from CSV file (default: yesterday to today, override with dates like ['2026-01-20','2026-01-22'])"
  task :backfill_ncaam, [:start_date, :end_date] => :environment do |t, args|
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : Date.current - 1.day
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : Date.current
    backfill_date_based_league('ncaam', start_date, end_date, 'db/data/ncaam_model_tracking.csv')
  end

  private

  def normalize_team_code(code)
    return code if code == 'Linden'
    code.upcase
  end

  def process_mlb_game_row(row, league, games_created, games_updated, results_created, predictions_created)
    # Parse date and teams
    game_date = Date.strptime(row['date'], '%Y%m%d')
    home_team_code = normalize_team_code(row['home_team'])
    away_team_code = normalize_team_code(row['away_team'])
    
    # Skip future games without moneyline data
    if game_date > Date.current && row['moneyline_favorite_team'].blank?
      puts "Skipping future game without moneyline: #{away_team_code} @ #{home_team_code} on #{game_date}"
      return [games_created, games_updated, results_created, predictions_created]
    end
    
    # Find teams
    home_team = league.teams.find_by!(code: home_team_code)
    away_team = league.teams.find_by!(code: away_team_code)
    
    # Find the season for this game date
    season = Season.find_by!(
      league: league,
      start_date: ..game_date,
      end_date: game_date..
    )
    
    # Parse start time or default to 1am of game date until time can be set
    start_time = if row['start_time'].present?
      Time.zone.parse(row['start_time']).utc
    else
      game_date.beginning_of_day.utc + 1.hour  # 1am UTC
    end
    
    # Determine initial status based on whether we have results
    has_results = row['away_result'].present? && row['home_result'].present?
    initial_status = has_results ? :final : :scheduled
    
    # Find or create game
    game = Game.find_or_create_by(
      league: league,
      start_time: start_time,
      home_team: home_team,
      away_team: away_team
    ) do |g|
      g.season = season
      g.status = initial_status
      games_created += 1
      status_text = has_results ? "final" : "scheduled"
      puts "Created: #{away_team_code} @ #{home_team_code} on #{game_date} (#{status_text})"
    end
    
    # Update start_time if we have better data and current time is 1am (placeholder)
    if !game.new_record? && row['start_time'].present? && game.start_time&.hour == 1
      game.update!(start_time: start_time)
      games_updated += 1
      puts "Updated time: #{away_team_code} @ #{home_team_code}"
    end
    
    # Create game result if final scores are present (skip if ESPN already provided final result)
    results_created += create_game_result(game, row, away_team_code, home_team_code)
    
    # FUTURE: Create game odds when MLB odds data is available
    # create_game_odds(game, row)
    
    # Create game predictions
    predictions_created += create_game_predictions(game, row, [
      { model: 'sl', away_col: 'sl_away_pred', home_col: 'sl_home_pred', source: 'sportsline' },
      { model: 'gdo_v6_pre', away_col: 'v6_pre_away', home_col: 'v6_pre_home', source: 'gdo' },
      { model: 'gdo_v8_pre', winner_col: 'v8_pre_winner', confidence_col: 'v8_pre_confidence', source: 'gdo' }
      # Commented out post-game predictions for future use:
      # { model: 'gdo_v1', away_col: 'v1_away', home_col: 'v1_home', source: 'gdo' },
      # { model: 'gdo_v2', away_col: 'v2_away', home_col: 'v2_home', source: 'gdo' },
      # { model: 'gdo_v5', away_col: 'v5_away', home_col: 'v5_home', source: 'gdo' },
      # { model: 'gdo_v6', away_col: 'v6_away', home_col: 'v6_home', source: 'gdo' },
      # { model: 'gdo_v8', winner_col: 'v8_winner', confidence_col: 'v8_confidence', source: 'gdo' }
    ])
    
    print "."
    [games_created, games_updated, results_created, predictions_created]
  end

  # def backfill_league_games(league_code, max_week, file_path)
  #   unless File.exist?(file_path)
  #     puts "Error: CSV file not found at #{file_path}"
  #     exit 1
  #   end
    
  #   league = League.find_by!(code: league_code)
  #   puts "Processing #{league_code.upcase} games from #{file_path}..."
  #   puts "Including games up to and including week #{max_week}"
    
  #   games_created = 0
  #   games_updated = 0
  #   results_created = 0
  #   games_skipped = 0
  #   odds_created = 0
  #   predictions_created = 0
    
  #   CSV.foreach(file_path, headers: true) do |row|
  #     begin
  #       result = process_week_game_row(row, league, max_week, games_created, games_updated, results_created, games_skipped, odds_created, predictions_created)
  #       games_created, games_updated, results_created, games_skipped, odds_created, predictions_created = result
  #     rescue => e
  #       puts "\nError processing row #{row['id']}: #{e.message}"
  #       puts "Row data: #{row.to_h}"
  #       raise e if Rails.env.development?
  #     end
  #   end
    
  #   puts "\n\n#{league_code.upcase} backfill complete!"
  #   puts "Games created: #{games_created}"
  #   puts "Games updated: #{games_updated}"
  #   puts "Results created: #{results_created}"
  #   puts "Odds created: #{odds_created}"
  #   puts "Predictions created: #{predictions_created}"
  #   puts "Games skipped: #{games_skipped}"
  #   puts "Total games in database: #{Game.count}"
  # end

  def backfill_date_based_league(league_code, start_date, end_date, file_path)
    unless File.exist?(file_path)
      puts "Error: CSV file not found at #{file_path}"
      exit 1
    end
    
    league = League.find_by!(code: league_code)
    puts "Processing #{league_code.upcase} games from #{file_path}..."
    puts "Including games from #{start_date} to #{end_date}"
    
    games_created = 0
    games_updated = 0
    results_created = 0
    games_skipped = 0
    odds_created = 0
    predictions_created = 0
    
    CSV.foreach(file_path, headers: true) do |row|
      begin
        result = process_date_game_row(row, league, start_date, end_date, games_created, games_updated, results_created, games_skipped, odds_created, predictions_created)
        games_created, games_updated, results_created, games_skipped, odds_created, predictions_created = result
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
    puts "Odds created: #{odds_created}"
    puts "Predictions created: #{predictions_created}"
    puts "Games skipped: #{games_skipped}"
    puts "Total games in database: #{Game.count}"
  end

  def process_date_game_row(row, league, start_date, end_date, games_created, games_updated, results_created, games_skipped, odds_created, predictions_created)
    # Parse basic game info
    game_date = Date.strptime(row['date'], '%Y%m%d')
    
    # Skip games outside the date range
    if game_date < start_date || game_date > end_date
      games_skipped += 1
      return [games_created, games_updated, results_created, games_skipped, odds_created, predictions_created]
    end
    
    home_team_code = normalize_team_code(row['home_team'])
    away_team_code = normalize_team_code(row['away_team'])
    
    # Find teams
    home_team = league.teams.find_by!(code: home_team_code)
    away_team = league.teams.find_by!(code: away_team_code)
    
    # Find the season for this game date
    season = Season.find_by!(
      league: league,
      start_date: ..game_date,
      end_date: game_date..
    )
    
    # Parse start time (MST format like "17:30")
    start_time = if row['start_time'].present?
      time_parts = row['start_time'].split(':')
      hour = time_parts[0].to_i
      minute = time_parts[1].to_i
      game_date.in_time_zone('America/Denver').change(hour: hour, min: minute).utc
    else
      game_date.beginning_of_day.utc + 1.hour
    end
    
    # Determine initial status based on whether we have results
    has_results = row['away_result'].present? && row['home_result'].present?
    initial_status = has_results ? :final : :scheduled
    
    # Find existing game(s) by identity (not start_time)
    existing_games = Game.where(league: league, home_team: home_team, away_team: away_team)
                         .for_date(game_date)

    game = if existing_games.count == 0
    Game.new(
      league: league,
      home_team: home_team,
      away_team: away_team
    )
    elsif existing_games.count == 1
      existing_games.first
    else
      # Doubleheader - find closest start time
      existing_games.min_by { |g| (g.start_time - start_time).abs }
    end

    is_new_game = game.new_record?

    if is_new_game
      game.assign_attributes(
        season: season,
        status: initial_status,
        start_time: start_time
      )
      game.save!
      games_created += 1
      status_text = has_results ? "final" : "scheduled"
      puts "Created: #{away_team_code} @ #{home_team_code} on #{game_date} (#{status_text})"
    else
      # Don't overwrite ESPN's start_time, only update status if needed
      game.update!(status: initial_status) if game.scheduled? && has_results
    end
    
    # Create game result if final scores are present (skip if ESPN already provided final result)
    results_created += create_game_result(game, row, away_team_code, home_team_code)
    
    # Create game odds from CSV only if no ESPN odds exist
    if game.game_odds.empty? && create_game_odds(game, row)
      odds_created += 1
    end
    
    # Create game predictions (NCAAM has sl and gdo_v1_pre)
    predictions_created += create_game_predictions(game, row, [
      { model: 'sl', away_col: 'sl_away_pred', home_col: 'sl_home_pred', source: 'sportsline' },
      { model: 'gdo_v1_pre', away_col: 'gdo_away_pred', home_col: 'gdo_home_pred', source: 'gdo' }
    ])
    
    print "."
    [games_created, games_updated, results_created, games_skipped, odds_created, predictions_created]
  end

  def create_game_odds(game, row)
    # Check if we have any odds data at all
    has_odds = row['moneyline_favorite_team'].present? ||
              row['runline_favorite_team'].present? ||
              row['total_line'].present?
    
    return false unless has_odds

    espn_source = DataSource.find_by!(code: 'espn')
    
    # Build odds attributes
    odds_attrs = {
      game: game,
      data_source: espn_source,
      fetched_at: Time.current
    }
    
    # Moneyline
    if row['moneyline_favorite_team'].present?
      moneyline_fav_team = game.league.teams.find_by!(code: normalize_team_code(row['moneyline_favorite_team']))
      odds_attrs[:moneyline_favorite_team] = moneyline_fav_team
      odds_attrs[:moneyline_favorite_odds] = row['favorite_moneyline_odds']&.to_i
      odds_attrs[:moneyline_underdog_odds] = row['underdog_moneyline_odds']&.to_i
    end

    # Spread/Runline
    if row['runline_favorite_team'].present? && row['runline_value'].present?
      spread_fav_team = game.league.teams.find_by!(code: normalize_team_code(row['runline_favorite_team']))
      odds_attrs[:spread_favorite_team] = spread_fav_team
      odds_attrs[:spread_value] = row['runline_value'].to_f
      odds_attrs[:spread_favorite_odds] = row['favorite_runline_odds']&.to_i
      odds_attrs[:spread_underdog_odds] = row['underdog_runline_odds']&.to_i
    end
    
    # Total (Over/Under)
    if row['total_line'].present?
      odds_attrs[:total_line] = row['total_line'].to_f
      odds_attrs[:over_odds] = row['over_total_line_odds']&.to_i
      odds_attrs[:under_odds] = row['under_total_line_odds']&.to_i
    end
    
    GameOdds.create!(odds_attrs)
    puts "  Odds created"
    true
  end

  def create_game_result(game, row, away_team_code, home_team_code)
    return 0 unless row['away_result'].present? && row['home_result'].present?
    
    existing_result = game.game_result
    
    if existing_result&.final?
      puts "  Result: skipped (ESPN final exists)"
      return 0
    end
    
    away_score = row['away_result'].to_i
    home_score = row['home_result'].to_i
    
    result = GameResult.find_or_initialize_by(game: game)
    result.assign_attributes(
      away_score: away_score,
      home_score: home_score,
      final: true
    )
    result.save!
    puts "  Result: #{away_team_code} #{away_score} - #{home_team_code} #{home_score}"
    
    game.final! if game.scheduled?
    
    1
  end

  def create_game_predictions(game, row, prediction_configs)
    count = 0
    
    prediction_configs.each do |config|
      model_version = config[:model]
      away_col = config[:away_col]
      home_col = config[:home_col]
      winner_col = config[:winner_col]
      confidence_col = config[:confidence_col]
      source_code = config[:source]

      # Skip if no prediction data present
      if away_col && home_col
        next if row[away_col].blank? && row[home_col].blank?
      elsif winner_col
        next if row[winner_col].blank?
      else
        next
      end

      # Find data source
      data_source = DataSource.find_by!(code: source_code)

      # Build prediction attributes
      prediction_attrs = {
        game: game,
        model_version: model_version,
        data_source: data_source,
        generated_at: game.start_time || game.game_date.beginning_of_day
      }

      # Add score predictions if present
      if away_col && home_col && row[away_col].present? && row[home_col].present?
        prediction_attrs[:away_predicted_score] = row[away_col].to_d
        prediction_attrs[:home_predicted_score] = row[home_col].to_d
      end

      # Add winner prediction if present
      if winner_col && row[winner_col].present?
        winner_team = game.league.teams.find_by!(code: normalize_team_code(row[winner_col]))
        prediction_attrs[:predicted_winner] = winner_team
        prediction_attrs[:confidence] = row[confidence_col]&.to_d if confidence_col
      end

      prediction = GamePrediction.find_or_create_by!(
        game: game,
        model_version: model_version,
        data_source: data_source,
        generated_at: prediction_attrs[:generated_at]
      ) do |p|
        p.away_predicted_score = prediction_attrs[:away_predicted_score]
        p.home_predicted_score = prediction_attrs[:home_predicted_score]
        p.predicted_winner = prediction_attrs[:predicted_winner]
        p.confidence = prediction_attrs[:confidence]
      end

      count += 1 if prediction.previously_new_record?
    end
    
    puts "  #{count} prediction(s) created" if count > 0
    count
  end
end