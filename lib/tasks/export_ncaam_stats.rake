namespace :export do
  desc "Export team stats to CSV"
  task :team_stats, [:league, :start_date, :end_date] => :environment do |t, args|
    league_code = args[:league] || 'ncaam'
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : Date.yesterday
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : end_date

    league = League.find_by!(code: league_code)
    
    start_time_begin = start_date.in_time_zone('America/Denver').beginning_of_day.utc
    start_time_end = end_date.in_time_zone('America/Denver').end_of_day.utc

    games = Game.joins(:basketball_game_team_stats, :game_result)
                .where(league: league)
                .where(start_time: start_time_begin..start_time_end)
                .where(game_results: { final: true })
                .includes(:home_team, :away_team, :game_result, :basketball_game_team_stats, :game_odds)
                .order(:start_time)

    dir = Rails.root.join("db/data/#{league_code}/raw")
    FileUtils.mkdir_p(dir)
    filepath = dir.join("team_stats.csv")

    headers = [
      'game_id', 'game_date', 'season',
      'team', 'opponent', 'home_away',
      'score', 'opponent_score', 'win',
      'field_goals_made', 'field_goals_attempted',
      'three_pointers_made', 'three_pointers_attempted',
      'free_throws_made', 'free_throws_attempted',
      'offensive_rebounds', 'defensive_rebounds',
      'assists', 'steals', 'blocks', 'turnovers', 'fouls',
      'points_off_turnovers', 'fast_break_points', 'points_in_paint',
      'largest_lead', 'lead_changes', 'time_leading_pct',
      'spread', 'spread_covered', 'total_line', 'went_over'
    ]

    CSV.open(filepath, 'w') do |csv|
      csv << headers

      games.each do |game|
        result = game.game_result
        current_odds = game.game_odds.where(is_opening: false).order(fetched_at: :desc).first
        
        game.basketball_game_team_stats.each do |stat|
          is_home = stat.team_id == game.home_team_id
          team = is_home ? game.home_team : game.away_team
          opponent = is_home ? game.away_team : game.home_team
          score = is_home ? result.home_score : result.away_score
          opponent_score = is_home ? result.away_score : result.home_score
          
          # Spread/total calculations
          spread = nil
          spread_covered = nil
          total_line = nil
          went_over = nil

          if current_odds
            total_line = current_odds.total_line
            game_total = result.home_score + result.away_score
            went_over = total_line ? game_total > total_line : nil

            if current_odds.spread_favorite_team_id == team.id
              # This team is the favorite, keep negative
              spread = current_odds.spread_value&.to_f
            elsif current_odds.spread_favorite_team_id == opponent.id
              # This team is the underdog, flip to positive
              spread = current_odds.spread_value&.to_f&.abs
            end
            
            spread_covered = spread ? (score + spread > opponent_score) : nil
          end

          csv << [
            game.id,
            game.game_date&.strftime('%Y-%m-%d'),
            game.season.name,
            team.code,
            opponent.code,
            is_home ? 'home' : 'away',
            score,
            opponent_score,
            score > opponent_score,
            stat.field_goals_made,
            stat.field_goals_attempted,
            stat.three_pointers_made,
            stat.three_pointers_attempted,
            stat.free_throws_made,
            stat.free_throws_attempted,
            stat.offensive_rebounds,
            stat.defensive_rebounds,
            stat.assists,
            stat.steals,
            stat.blocks,
            stat.turnovers,
            stat.fouls,
            stat.points_off_turnovers,
            stat.fast_break_points,
            stat.points_in_paint,
            stat.largest_lead,
            stat.lead_changes,
            stat.time_leading_pct,
            spread,
            spread_covered,
            total_line,
            went_over
          ]
        end
      end
    end

    puts "Exported #{games.count} games (#{games.count * 2} team rows) to #{filepath}"
  end

  desc "Export player stats to CSV"
  task :player_stats, [:league, :start_date, :end_date] => :environment do |t, args|
    league_code = args[:league] || 'ncaam'
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : Date.yesterday
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : end_date

    league = League.find_by!(code: league_code)

    start_time_begin = start_date.in_time_zone('America/Denver').beginning_of_day.utc
    start_time_end = end_date.in_time_zone('America/Denver').end_of_day.utc

    game_ids = Game.joins(:basketball_game_player_stats, :game_result)
              .where(league: league)
              .where(start_time: start_time_begin..start_time_end)
              .where(game_results: { final: true })
              .pluck(:id)

    games = Game.where(id: game_ids)
                .includes(:home_team, :away_team, :game_result, basketball_game_player_stats: :player)
                .order(:start_time)

    dir = Rails.root.join("db/data/#{league_code}/raw")
    FileUtils.mkdir_p(dir)
    filepath = dir.join("player_stats.csv")

    headers = [
      'game_id', 'game_date', 'season',
      'player_id', 'player_name', 'position',
      'team', 'opponent', 'home_away',
      'team_score', 'opponent_score', 'team_win',
      'minutes_played', 'minutes_pct',
      'points',
      'field_goals_made', 'field_goals_attempted',
      'three_pointers_made', 'three_pointers_attempted',
      'free_throws_made', 'free_throws_attempted',
      'offensive_rebounds', 'defensive_rebounds',
      'assists', 'steals', 'blocks', 'turnovers', 'fouls'
    ]

    player_stat_count = 0


    CSV.open(filepath, 'w') do |csv|
      csv << headers

      games.each do |game|
        result = game.game_result
        
        # Calculate total game minutes (40 + 5 per OT)
        period = result.period || 2
        total_minutes = period <= 2 ? 40 : 40 + ((period - 2) * 5)

        game.basketball_game_player_stats.each do |stat|
          is_home = stat.team_id == game.home_team_id
          team = is_home ? game.home_team : game.away_team
          opponent = is_home ? game.away_team : game.home_team
          team_score = is_home ? result.home_score : result.away_score
          opponent_score = is_home ? result.away_score : result.home_score

          minutes_pct = stat.minutes_played ? (stat.minutes_played.to_f / total_minutes).round(3) : nil

          csv << [
            game.id,
            game.game_date&.strftime('%Y-%m-%d'),
            game.season.name,
            stat.player_id,
            stat.player.name,
            stat.player.position,
            team.code,
            opponent.code,
            is_home ? 'home' : 'away',
            team_score,
            opponent_score,
            team_score > opponent_score,
            stat.minutes_played,
            minutes_pct,
            stat.points,
            stat.field_goals_made,
            stat.field_goals_attempted,
            stat.three_pointers_made,
            stat.three_pointers_attempted,
            stat.free_throws_made,
            stat.free_throws_attempted,
            stat.offensive_rebounds,
            stat.defensive_rebounds,
            stat.assists,
            stat.steals,
            stat.blocks,
            stat.turnovers,
            stat.fouls
          ]

          player_stat_count += 1
        end
      end
    end

    puts "Exported #{player_stat_count} player stat rows from #{games.count} games to #{filepath}"
  end
end