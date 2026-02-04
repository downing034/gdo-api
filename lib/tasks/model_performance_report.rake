namespace :models do
  desc "Compare model performance (usage: rails models:performance_report[ncaam,2026-01-24] or rails models:performance_report[ncaam,2026-01-20,2026-01-24])"
  task :performance_report, [:league_code, :start_date, :end_date] => :environment do |t, args|
    league_code = args[:league_code] || 'ncaam'
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : Date.yesterday
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : start_date
    
    league = League.find_by!(code: league_code)
    
    start_time_begin = start_date.in_time_zone('America/Denver').beginning_of_day.utc
    start_time_end = end_date.in_time_zone('America/Denver').end_of_day.utc

    games = Game.includes(:home_team, :away_team, :game_result, :game_predictions, :game_odds)
                .where(league: league)
                .where(start_time: start_time_begin..start_time_end)
                .where(status: 'final')
    
    if games.empty?
      puts "No final games with results for #{league_code.upcase} #{start_date}#{end_date != start_date ? " to #{end_date}" : ""}"
      exit 0
    end
    
    # Models that predict scores (can evaluate ML, ATS, O/U, MAE)
    score_models = ['v1', 'v2_vegas', 'v2_no_vegas', 'v3', 'sl']
    
    # Models that predict winner only (can only evaluate ML)
    winner_only_models = ['v4']
    
    all_prediction_models = score_models + winner_only_models
    
    espn_source = DataSource.find_by!(code: 'espn')
    
    # Model stats
    stats = all_prediction_models.each_with_object({}) do |model, h|
      h[model] = {
        games: 0,
        ml_correct: 0,
        ats_correct: 0,
        ats_push: 0,
        ou_over_correct: 0,
        ou_under_correct: 0,
        ou_push: 0,
        total_error_away: 0,
        total_error_home: 0,
        games_with_odds: 0,
        games_with_ou: 0,
        winner_only: winner_only_models.include?(model)
      }
    end
    
    # Combined v2 stats (best of vegas or no_vegas for each game)
    stats['v2_combined'] = {
      games: 0,
      ml_correct: 0,
      ats_correct: 0,
      ats_push: 0,
      ou_over_correct: 0,
      ou_under_correct: 0,
      ou_push: 0,
      total_error_away: 0,
      total_error_home: 0,
      games_with_odds: 0,
      games_with_ou: 0,
      winner_only: false
    }
    
    # Oddsmaker stats
    odds_stats = {
      ml_games: 0,
      ml_correct: 0,
      ats_games: 0,
      ats_correct: 0,
      ats_push: 0
    }
    
    games.each do |game|
      result = game.game_result
      actual_home = result.home_score
      actual_away = result.away_score
      actual_winner = actual_home > actual_away ? game.home_team : game.away_team
      actual_total = actual_home + actual_away
      
      odds = game.game_odds
                 .where(data_source: espn_source, is_opening: false)
                 .order(fetched_at: :desc)
                 .first
      
      # Oddsmaker moneyline
      if odds&.moneyline_favorite_team
        odds_stats[:ml_games] += 1
        odds_stats[:ml_correct] += 1 if odds.moneyline_favorite_team == actual_winner
      end
      
      # Oddsmaker ATS (did the favorite cover?)
      if odds&.spread_favorite_team && odds&.spread_value
        odds_stats[:ats_games] += 1
        spread_value = odds.spread_value.to_f.abs
        favorite = odds.spread_favorite_team
        
        favorite_score = favorite == game.home_team ? actual_home : actual_away
        underdog_score = favorite == game.home_team ? actual_away : actual_home
        
        margin = favorite_score - underdog_score
        
        if margin == spread_value
          odds_stats[:ats_push] += 1
        elsif margin > spread_value
          odds_stats[:ats_correct] += 1
        end
      end
      
      # Find v2 combined prediction (prefer vegas, fallback to no_vegas)
      v2_combined_pred = game.game_predictions.find { |p| p.model_version == 'v2_vegas' } ||
                         game.game_predictions.find { |p| p.model_version == 'v2_no_vegas' }
      
      (all_prediction_models + ['v2_combined']).each do |model|
        pred = if model == 'v2_combined'
          v2_combined_pred
        else
          game.game_predictions.find { |p| p.model_version == model }
        end
        next unless pred
        
        s = stats[model]
        s[:games] += 1
        
        is_winner_only = s[:winner_only]
        
        # For winner-only models, use predicted_winner directly
        # For score models, derive winner from scores
        if is_winner_only
          pred_winner = pred.predicted_winner
        else
          pred_home = pred.home_predicted_score.to_f
          pred_away = pred.away_predicted_score.to_f
          pred_winner = pred_home > pred_away ? game.home_team : game.away_team
          pred_total = pred_home + pred_away
        end
        
        # Moneyline (all models)
        s[:ml_correct] += 1 if pred_winner == actual_winner
        
        # Skip ATS, O/U, MAE for winner-only models
        next if is_winner_only
        
        # MAE (score models only)
        s[:total_error_away] += (pred_away - actual_away).abs
        s[:total_error_home] += (pred_home - actual_home).abs
        
        next unless odds
        
        s[:games_with_odds] += 1
        
        # ATS (score models only)
        if odds.spread_favorite_team && odds.spread_value
          spread_value = odds.spread_value.to_f.abs
          favorite = odds.spread_favorite_team
          
          favorite_score = favorite == game.home_team ? actual_home : actual_away
          underdog_score = favorite == game.home_team ? actual_away : actual_home
          actual_margin = favorite_score - underdog_score
          
          pred_favorite_score = favorite == game.home_team ? pred_home : pred_away
          pred_underdog_score = favorite == game.home_team ? pred_away : pred_home
          pred_margin = pred_favorite_score - pred_underdog_score
          
          pred_favorite_covers = pred_margin > spread_value
          actual_favorite_covers = actual_margin > spread_value
          
          if actual_margin == spread_value
            s[:ats_push] += 1
          elsif pred_favorite_covers == actual_favorite_covers
            s[:ats_correct] += 1
          end
        end
        
        # Over/Under (score models only)
        if odds.total_line
          s[:games_with_ou] += 1
          total_line = odds.total_line.to_f
          pred_over = pred_total > total_line
          actual_over = actual_total > total_line
          
          if actual_total == total_line
            s[:ou_push] += 1
          elsif pred_over && actual_over
            s[:ou_over_correct] += 1
          elsif !pred_over && !actual_over
            s[:ou_under_correct] += 1
          end
        end
      end
    end
    
    # Output
    all_models_display = ['v1', 'v2_combined', 'v2_no_vegas', 'v3', 'v4', 'sl']
    date_range = start_date == end_date ? start_date.to_s : "#{start_date} to #{end_date}"
    puts "=" * 60
    puts "Model Performance: #{league_code.upcase} #{date_range} (#{games.count} games)"
    puts "=" * 60
    
    # Moneyline (all models)
    puts "\nMONEYLINE"
    puts "-" * 40
    if odds_stats[:ml_games] > 0
      ml_pct = (odds_stats[:ml_correct].to_f / odds_stats[:ml_games] * 100).round(1)
      puts "Oddsmaker:      #{odds_stats[:ml_correct]}/#{odds_stats[:ml_games]} (#{ml_pct}%)"
    end
    all_models_display.each do |model|
      s = stats[model]
      next if s.nil? || s[:games] == 0
      ml_pct = (s[:ml_correct].to_f / s[:games] * 100).round(1)
      label = model.upcase.gsub('_', ' ')
      puts "#{label.ljust(14)}#{s[:ml_correct]}/#{s[:games]} (#{ml_pct}%)"
    end
    
    # ATS (score models only)
    puts "\nATS"
    puts "-" * 40
    if odds_stats[:ats_games] > 0
      ats_decided = odds_stats[:ats_games] - odds_stats[:ats_push]
      ats_pct = ats_decided > 0 ? (odds_stats[:ats_correct].to_f / ats_decided * 100).round(1) : 0
      puts "Oddsmaker:      #{odds_stats[:ats_correct]}/#{ats_decided} (#{ats_pct}%) [#{odds_stats[:ats_push]} push]"
    end
    all_models_display.each do |model|
      s = stats[model]
      next if s.nil? || s[:games_with_odds] == 0 || s[:winner_only]
      ats_decided = s[:games_with_odds] - s[:ats_push]
      ats_pct = ats_decided > 0 ? (s[:ats_correct].to_f / ats_decided * 100).round(1) : 0
      label = model.upcase.gsub('_', ' ')
      puts "#{label.ljust(14)}#{s[:ats_correct]}/#{ats_decided} (#{ats_pct}%) [#{s[:ats_push]} push]"
    end
    
    # Over/Under (score models only)
    puts "\nOVER/UNDER"
    puts "-" * 40
    all_models_display.each do |model|
      s = stats[model]
      next if s.nil? || s[:games_with_ou] == 0 || s[:winner_only]
      ou_correct = s[:ou_over_correct] + s[:ou_under_correct]
      ou_decided = s[:games_with_ou] - s[:ou_push]
      ou_pct = ou_decided > 0 ? (ou_correct.to_f / ou_decided * 100).round(1) : 0
      label = model.upcase.gsub('_', ' ')
      puts "#{label.ljust(14)}#{ou_correct}/#{ou_decided} (#{ou_pct}%) [#{s[:ou_push]} push]"
    end
    
    # MAE (score models only)
    puts "\nMAE"
    puts "-" * 40
    all_models_display.each do |model|
      s = stats[model]
      next if s.nil? || s[:games] == 0 || s[:winner_only]
      mae_away = (s[:total_error_away] / s[:games]).round(1)
      mae_home = (s[:total_error_home] / s[:games]).round(1)
      mae_avg = ((mae_away + mae_home) / 2).round(1)
      label = model.upcase.gsub('_', ' ')
      puts "#{label.ljust(14)}#{mae_avg} (away: #{mae_away}, home: #{mae_home})"
    end
    
    puts "\n"
  end
end