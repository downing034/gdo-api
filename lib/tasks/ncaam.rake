namespace :ncaam do
  desc "Process raw CSVs into model-ready data"
  task process: :environment do
    puts "=" * 60
    puts "Processing raw data..."
    puts "=" * 60
    
    # Check for SEASONS env var, default to current season only
    seasons = if ENV['SEASONS']
      ENV['SEASONS'].split(',').map(&:strip)
    else
      ['25_26']
    end
    
    puts "Seasons: #{seasons.join(', ')}"
    
    Ncaam::DataProcessorService.new(seasons: seasons).call
  end

  desc "Train the prediction model"
  task train: :environment do
    venv_python = Rails.root.join('db', 'data', 'ncaam', 'venv', 'bin', 'python')
    data_dir = Rails.root.join('db', 'data', 'ncaam')
    
    # Train v1
    puts "=" * 60
    puts "Training v1 model..."
    puts "=" * 60
    
    v1_script = Rails.root.join('db', 'data', 'ncaam', 'models', 'v1', 'train_model.py')
    system("#{venv_python} #{v1_script}") || raise("V1 training failed")
    
    # Train v2
    puts
    puts "=" * 60
    puts "Training v2 models (vegas + no-vegas)..."
    puts "=" * 60
    
    v2_script = data_dir.join('models', 'v2', 'train.py')
    v2_cmd = [
      venv_python.to_s,
      v2_script.to_s,
      "--bart-games", data_dir.join('processed', 'base_model_game_data_with_rolling.csv').to_s,
      "--bart-season", data_dir.join('processed', 'ncaam_team_data_final.csv').to_s,
      "--espn-team", data_dir.join('raw', 'team_stats.csv').to_s,
      "--espn-player", data_dir.join('raw', 'player_stats.csv').to_s,
      "--output-dir", data_dir.join('models', 'v2').to_s
    ].join(' ')
    
    system(v2_cmd) || raise("V2 training failed")
    
    # Train v3 (score prediction model)
    puts
    puts "=" * 60
    puts "Training v3 model (dual score prediction)..."
    puts "=" * 60
    
    v3_script = Rails.root.join('db', 'data', 'ncaam', 'models', 'v3', 'train.py')
    system("#{venv_python} #{v3_script}") || raise("V3 training failed")
    
    # Train v4 (winner-only model)
    puts
    puts "=" * 60
    puts "Training v4 model (winner-only, moneyline optimized)..."
    puts "=" * 60
    
    v4_script = Rails.root.join('db', 'data', 'ncaam', 'models', 'v4', 'train.py')
    system("#{venv_python} #{v4_script}") || raise("V4 training failed")
    
    puts
    puts "=" * 60
    puts "Training complete!"
    puts "=" * 60
  end

  desc "Generate predictions for upcoming games (optional: date or date range, include_completed_games=true to include final games)"
  task :predict, [:start_date, :end_date, :include_completed_games] => :environment do |t, args|
    start_date = args[:start_date] ? Date.parse(args[:start_date]) : nil
    end_date = args[:end_date] ? Date.parse(args[:end_date]) : nil
    include_completed_games = args[:include_completed_games] == 'true'
    
    %w[v1 v2 v3 v4].each do |version|
      puts "\n=== Running #{version} predictions ==="
      results = Ncaam::PredictService.new(
        model_version: version,
        start_date: start_date, 
        end_date: end_date, 
        include_completed_games: include_completed_games
      ).call
      
      puts "Created: #{results[:created]}"
      puts "Updated: #{results[:updated]}"
      puts "Skipped: #{results[:skipped]}"
      puts "Errors: #{results[:errors].count}"
      
      if results[:errors].any?
        puts "\nErrors:"
        results[:errors].each { |e| puts "  #{e}" }
      end
    end
  end
    
  desc "Process data and train model (run after dropping new CSVs)"
  task refresh: :environment do
    results = Ncaam::RefreshService.new.call
    
    puts "Processed: #{results[:processed]}"
    puts "Trained: #{results[:trained]}"
  end

  desc "Analyze NCAAM sweet spot picks. Options: [today_date, yesterday_date, stats_start, stats_end]"
  task :sweet_spot, [:today_date, :yesterday_date, :stats_start, :stats_end] => :environment do |t, args|
    today = args[:today_date] ? Date.parse(args[:today_date]) : Date.current
    yesterday = args[:yesterday_date] ? Date.parse(args[:yesterday_date]) : today - 1.day
    stats_start = args[:stats_start] ? Date.parse(args[:stats_start]) : Date.new(2026, 1, 2)
    stats_end = args[:stats_end] ? Date.parse(args[:stats_end]) : yesterday

    analyzer = Ncaam::SweetSpotAnalyzer.new(league_code: 'ncaam')

    # Gather all data upfront
    yesterday_points = analyzer.sweet_spot_games(yesterday)
    yesterday_ml = analyzer.v4_sweet_spot_games(yesterday)
    yesterday_dogs = analyzer.v4_underdog_picks(yesterday)
    
    today_points = analyzer.sweet_spot_games(today)
    today_ml = analyzer.v4_sweet_spot_games(today)
    today_dogs = analyzer.v4_underdog_picks(today)
    today_combined = analyzer.combined_sweet_spot_games(today)

    points_stats = analyzer.accuracy_stats(start_date: stats_start, end_date: stats_end)
    ml_stats = analyzer.v4_accuracy_stats(start_date: stats_start, end_date: stats_end)

    # =========================================================================
    # YESTERDAY'S RESULTS
    # =========================================================================
    puts "=" * 80
    puts "📊 YESTERDAY'S RESULTS (#{yesterday})"
    puts "=" * 80

    # Points Model Results
    puts "\n🏀 POINTS MODEL (V3+SL agree, pick fav, margin≥8)"
    puts "-" * 80
    if yesterday_points.empty?
      puts "   No games"
    else
      yesterday_points.sort_by { |a| a[:tier] == 2.0 ? 0 : 1 }.each do |a|
        game = a[:game]
        result = game.game_result
        icon = a[:gdo_correct] ? "✓" : "✗"
        tier = a[:tier] == 2.0 ? "2.0" : "1.0"
        score = result&.final? ? "#{result.away_score}-#{result.home_score}" : "pending"
        margin = result&.final? ? (result.away_score - result.home_score).abs : 0
        puts "   #{icon} [#{tier}] #{a[:gdo_pick].code.ljust(5)} spread:#{a[:spread_size].to_s.rjust(5)}  margin:#{a[:gdo_margin].round(0).to_s.rjust(3)}  |  #{score} (#{margin})"
      end
    end

    # Moneyline Model Results
    puts "\n💰 MONEYLINE MODEL (V4 conf≥70/80%, spread≥13, pick fav)"
    puts "-" * 80
    if yesterday_ml.empty?
      puts "   No games"
    else
      yesterday_ml.sort_by { |a| a[:v4_tier] == 2.0 ? 0 : 1 }.each do |a|
        game = a[:game]
        result = game.game_result
        icon = a[:v4_correct] ? "✓" : "✗"
        tier = a[:v4_tier] == 2.0 ? "2.0" : "1.0"
        score = result&.final? ? "#{result.away_score}-#{result.home_score}" : "pending"
        puts "   #{icon} [#{tier}] #{a[:v4_pick]&.code.to_s.ljust(5)} spread:#{a[:spread_size].to_s.rjust(5)}  conf:#{a[:v4_confidence]&.round(0).to_s.rjust(3)}%  |  #{score}"
      end
    end

    # Underdog Results
    puts "\n🐕 UNDERDOGS (V4 conf≥60%, picking underdog)"
    puts "-" * 80
    if yesterday_dogs.empty?
      puts "   No games"
    else
      yesterday_dogs.each do |a|
        game = a[:game]
        result = game.game_result
        icon = a[:v4_correct] ? "✓" : "✗"
        score = result&.final? ? "#{result.away_score}-#{result.home_score}" : "pending"
        puts "   #{icon}      #{a[:v4_pick]&.code.to_s.ljust(5)} spread:#{a[:spread_size].to_s.rjust(5)}  conf:#{a[:v4_confidence]&.round(0).to_s.rjust(3)}%  |  #{score}"
      end
    end

    # =========================================================================
    # ACCURACY STATS
    # =========================================================================
    puts "\n\n" + "=" * 80
    puts "📈 ACCURACY STATS (#{stats_start} to #{stats_end})"
    puts "=" * 80

    puts "\n🏀 POINTS MODEL"
    puts "-" * 50
    puts "   #{'Spread'.ljust(12)} #{'Record'.rjust(12)} #{'Pct'.rjust(8)}"
    points_stats.each do |stat|
      puts "   ≥#{stat[:threshold].to_s.ljust(10)} #{stat[:correct].to_s.rjust(5)}/#{stat[:total].to_s.ljust(5)} #{(stat[:accuracy].to_s + '%').rjust(8)}"
    end

    puts "\n💰 MONEYLINE MODEL"
    puts "-" * 50
    ml_stats.each do |stat|
      short_name = stat[:name].gsub('V4 ', '').gsub(' (conf>=', ' ≥').gsub(', spread>=', ' spd≥').gsub(', fav)', ' fav').gsub(')', '')
      puts "   #{short_name.ljust(28)} #{stat[:correct].to_s.rjust(4)}/#{stat[:total].to_s.ljust(4)} #{(stat[:accuracy].to_s + '%').rjust(7)}"
    end

    # =========================================================================
    # TODAY'S PICKS
    # =========================================================================
    puts "\n\n" + "=" * 80
    puts "🎯 TODAY'S PICKS (#{today})"
    puts "=" * 80

    # Get accuracy for combined (use the higher threshold - SS2.0 spread>=13)
    combined_stat = points_stats.find { |s| s[:threshold] == 13 }
    ml_ss2_stat = ml_stats.find { |s| s[:name].include?('SS 2.0') }
    ml_ss1_stat = ml_stats.find { |s| s[:name].include?('SS 1.0') }
    dog_stat = ml_stats.find { |s| s[:name].include?('underdog') }

    # Combined picks (both models agree) - THE BEST
    puts "\n🔥 COMBINED (Both Models) #{combined_stat ? "#{combined_stat[:correct]}/#{combined_stat[:total]} (#{combined_stat[:accuracy]}%)" : ''}"
    puts "-" * 80
    if today_combined.empty?
      puts "   No games qualify for both models"
    else
      puts "   #{'Time'.ljust(6)} #{'Game'.ljust(14)} #{'Pick'.ljust(6)} #{'Spread'.rjust(7)} #{'Conf'.rjust(6)} #{'Margin'.rjust(7)} #{'ML Odds'.rjust(8)}"
      puts "   " + "-" * 60
      today_combined.sort_by { |a| a[:game].start_time }.each do |a|
        game = a[:game]
        time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        ml_odds = a[:moneyline_favorite_odds] || "-"
        puts "   #{time.ljust(6)} #{matchup.ljust(14)} #{a[:gdo_pick].code.ljust(6)} #{a[:spread_size].to_s.rjust(7)} #{(a[:v4_confidence]&.round(0).to_s + '%').rjust(6)} #{a[:gdo_margin].round(0).to_s.rjust(7)} #{ml_odds.to_s.rjust(8)}"
      end
    end

    # Points Model Only (not in combined)
    points_only = today_points.reject { |a| today_combined.any? { |c| c[:game].id == a[:game].id } }
    points_1_stat = points_stats.find { |s| s[:threshold] == 10 }
    puts "\n🏀 POINTS MODEL ONLY — 2.0: #{combined_stat ? "#{combined_stat[:accuracy]}%" : '-'} | 1.0: #{points_1_stat ? "#{points_1_stat[:accuracy]}%" : '-'}"
    puts "-" * 80
    if points_only.empty?
      puts "   No additional games"
    else
      puts "   #{'Time'.ljust(6)} #{'Game'.ljust(14)} #{'Tier'.ljust(5)} #{'Pick'.ljust(6)} #{'Spread'.rjust(7)} #{'Margin'.rjust(7)} #{'ML Odds'.rjust(8)}"
      puts "   " + "-" * 60
      points_only.sort_by { |a| a[:game].start_time }.each do |a|
        game = a[:game]
        time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        tier = a[:tier] == 2.0 ? "2.0" : "1.0"
        ml_odds = a[:moneyline_favorite_odds] || "-"
        puts "   #{time.ljust(6)} #{matchup.ljust(14)} #{tier.ljust(5)} #{a[:gdo_pick].code.ljust(6)} #{a[:spread_size].to_s.rjust(7)} #{a[:gdo_margin].round(0).to_s.rjust(7)} #{ml_odds.to_s.rjust(8)}"
      end
    end

    # Moneyline Model Only (not in combined)
    ml_only = today_ml.reject { |a| today_combined.any? { |c| c[:game].id == a[:game].id } }
    puts "\n💰 MONEYLINE MODEL ONLY — 2.0: #{ml_ss2_stat ? "#{ml_ss2_stat[:accuracy]}%" : '-'} | 1.0: #{ml_ss1_stat ? "#{ml_ss1_stat[:accuracy]}%" : '-'}"
    puts "-" * 80
    if ml_only.empty?
      puts "   No additional games"
    else
      puts "   #{'Time'.ljust(6)} #{'Game'.ljust(14)} #{'Tier'.ljust(5)} #{'Pick'.ljust(6)} #{'Spread'.rjust(7)} #{'Conf'.rjust(6)} #{'ML Odds'.rjust(8)}"
      puts "   " + "-" * 60
      ml_only.sort_by { |a| a[:game].start_time }.each do |a|
        game = a[:game]
        time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        tier = a[:v4_tier] == 2.0 ? "2.0" : "1.0"
        ml_odds = a[:moneyline_favorite_odds] || "-"
        puts "   #{time.ljust(6)} #{matchup.ljust(14)} #{tier.ljust(5)} #{a[:v4_pick]&.code.to_s.ljust(6)} #{a[:spread_size].to_s.rjust(7)} #{(a[:v4_confidence]&.round(0).to_s + '%').rjust(6)} #{ml_odds.to_s.rjust(8)}"
      end
    end

    # Underdog Picks
    puts "\n🐕 UNDERDOG PICKS — #{dog_stat ? "#{dog_stat[:correct]}/#{dog_stat[:total]} (#{dog_stat[:accuracy]}%)" : ''}"
    puts "-" * 80
    if today_dogs.empty?
      puts "   No underdog picks"
    else
      puts "   #{'Time'.ljust(6)} #{'Game'.ljust(14)} #{'Pick'.ljust(6)} #{'Spread'.rjust(7)} #{'Conf'.rjust(6)} #{'ML Odds'.rjust(8)}"
      puts "   " + "-" * 55
      today_dogs.sort_by { |a| a[:game].start_time }.each do |a|
        game = a[:game]
        time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        ml_odds = a[:moneyline_underdog_odds] || "-"
        puts "   #{time.ljust(6)} #{matchup.ljust(14)} #{a[:v4_pick]&.code.to_s.ljust(6)} #{a[:spread_size].to_s.rjust(7)} #{(a[:v4_confidence]&.round(0).to_s + '%').rjust(6)} #{ml_odds.to_s.rjust(8)}"
      end
    end

    # =========================================================================
    # PARLAY RANKINGS
    # =========================================================================
    puts "\n\n" + "=" * 80
    puts "🎰 PARLAY RANKINGS (#{today})"
    puts "=" * 80

    puts "\n🔥 COMBINED (safest - both models agree)"
    puts "-" * 60
    if today_combined.empty?
      puts "   No games"
    else
      today_combined.sort_by { |a| -a[:spread_size] }.each_with_index do |a, i|
        ml_odds = a[:moneyline_favorite_odds] || "-"
        puts "   #{(i+1).to_s.rjust(2)}. #{a[:gdo_pick].code.ljust(6)} spd:#{a[:spread_size].to_s.rjust(5)}  conf:#{a[:v4_confidence]&.round(0).to_s.rjust(3)}%  margin:#{a[:gdo_margin].round(0).to_s.rjust(3)}  #{ml_odds}"
      end
    end

    puts "\n🏀 POINTS MODEL"
    puts "-" * 60
    rankings = analyzer.parlay_rankings(today)
    if rankings.empty?
      puts "   No games"
    else
      rankings.each_with_index do |a, i|
        tier = a[:tier] == 2.0 ? "2.0" : "1.0"
        ml_odds = a[:moneyline_favorite_odds] || "-"
        puts "   #{(i+1).to_s.rjust(2)}. [#{tier}] #{a[:gdo_pick].code.ljust(6)} spd:#{a[:spread_size].to_s.rjust(5)}  margin:#{a[:gdo_margin].round(0).to_s.rjust(3)}  #{ml_odds}"
      end
    end

    puts "\n💰 MONEYLINE MODEL"
    puts "-" * 60
    v4_rankings = analyzer.v4_parlay_rankings(today)
    if v4_rankings.empty?
      puts "   No games"
    else
      v4_rankings.each_with_index do |a, i|
        tier = a[:v4_tier] == 2.0 ? "2.0" : "1.0"
        ml_odds = a[:moneyline_favorite_odds] || "-"
        puts "   #{(i+1).to_s.rjust(2)}. [#{tier}] #{a[:v4_pick]&.code.to_s.ljust(6)} spd:#{a[:spread_size].to_s.rjust(5)}  conf:#{a[:v4_confidence]&.round(0).to_s.rjust(3)}%  #{ml_odds}"
      end
    end

    puts "\n" + "=" * 80
  end
end