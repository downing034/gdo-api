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
    puts "=" * 60
    puts "Training model..."
    puts "=" * 60
    
    venv_python = Rails.root.join('db', 'data', 'ncaam', 'venv', 'bin', 'python')
    train_script = Rails.root.join('db', 'data', 'ncaam', 'models', 'v1', 'train_model.py')
    
    system("#{venv_python} #{train_script}") || raise("Training failed")
  end

  desc "Generate predictions for upcoming games"
  task predict: :environment do
    results = Ncaam::PredictService.new.call
    
    puts "Created: #{results[:created]}"
    puts "Updated: #{results[:updated]}"
    puts "Skipped: #{results[:skipped]}"
    puts "Errors: #{results[:errors].count}"
    
    if results[:errors].any?
      puts "\nErrors:"
      results[:errors].each { |e| puts "  #{e}" }
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

    # Yesterday's Results
    puts "=" * 70
    puts "📊 YESTERDAY'S SWEET SPOT RESULTS (#{yesterday})"
    puts "=" * 70

    yesterday_picks = analyzer.sweet_spot_games(yesterday)
    
    if yesterday_picks.empty?
      puts "No sweet spot games yesterday."
    else
      yesterday_picks.sort_by { |a| a[:game].start_time }.each do |a|
        game = a[:game]
        result = game.game_result
        
        icon = a[:gdo_correct] ? "✓" : "✗"
        tier_label = a[:tier] == 2.0 ? "SS2.0" : "SS1.0"
        
        if result&.final?
          score = "#{game.away_team.code} #{result.away_score} - #{game.home_team.code} #{result.home_score}"
          winner_score = [result.away_score, result.home_score].max
          loser_score = [result.away_score, result.home_score].min
          win_margin = winner_score - loser_score
        else
          score = "No result"
          win_margin = "-"
        end

        puts "#{icon} [#{tier_label}] #{a[:gdo_pick].code} (spread: #{a[:spread_size]}, margin: #{a[:gdo_margin].round(1)}) | #{score} | Win by: #{win_margin}"
      end
    end

    # Accuracy Stats
    puts "\n" + "=" * 70
    puts "📈 OVERALL ACCURACY STATS (#{stats_start} to #{stats_end})"
    puts "=" * 70
    puts "#{'Spread ≥'.ljust(12)} #{'Correct'.rjust(8)} #{'Total'.rjust(8)} #{'Accuracy'.rjust(10)}"
    puts "-" * 40

    analyzer.accuracy_stats(start_date: stats_start, end_date: stats_end).each do |stat|
      puts "#{stat[:threshold].to_s.ljust(12)} #{stat[:correct].to_s.rjust(8)} #{stat[:total].to_s.rjust(8)} #{(stat[:accuracy].to_s + '%').rjust(10)}"
    end

    # Today's Picks - Sweet Spot 2.0
    puts "\n" + "=" * 70
    puts "🏆 TODAY'S SWEET SPOT 2.0 PICKS (#{today})"
    puts "=" * 70

    today_picks = analyzer.sweet_spot_games(today)
    ss2_picks = today_picks.select { |a| a[:tier] == 2.0 }.sort_by { |a| a[:game].start_time }

    if ss2_picks.empty?
      puts "No Sweet Spot 2.0 picks today."
    else
      puts "#{'Time'.ljust(8)} #{'Game'.ljust(20)} #{'Pick'.ljust(8)} #{'Spread'.rjust(8)} #{'ML Odds'.rjust(10)} #{'GDO Margin'.rjust(12)}"
      puts "-" * 70
      
      ss2_picks.each do |a|
        game = a[:game]
        time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        ml_odds = a[:moneyline_favorite_odds] ? a[:moneyline_favorite_odds].to_s : "-"
        
        puts "#{time.ljust(8)} #{matchup.ljust(20)} #{a[:gdo_pick].code.ljust(8)} #{a[:spread_size].to_s.rjust(8)} #{ml_odds.rjust(10)} #{a[:gdo_margin].round(1).to_s.rjust(12)}"
      end
    end

    # Today's Picks - Sweet Spot 1.0
    puts "\n" + "=" * 70
    puts "⚠️  TODAY'S SWEET SPOT 1.0 PICKS (#{today})"
    puts "=" * 70

    ss1_picks = today_picks.select { |a| a[:tier] == 1.0 }.sort_by { |a| a[:game].start_time }

    if ss1_picks.empty?
      puts "No Sweet Spot 1.0 picks today."
    else
      puts "#{'Time'.ljust(8)} #{'Game'.ljust(20)} #{'Pick'.ljust(8)} #{'Spread'.rjust(8)} #{'ML Odds'.rjust(10)} #{'GDO Margin'.rjust(12)}"
      puts "-" * 70
      
      ss1_picks.each do |a|
        game = a[:game]
        time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        ml_odds = a[:moneyline_favorite_odds] ? a[:moneyline_favorite_odds].to_s : "-"
        
        puts "#{time.ljust(8)} #{matchup.ljust(20)} #{a[:gdo_pick].code.ljust(8)} #{a[:spread_size].to_s.rjust(8)} #{ml_odds.rjust(10)} #{a[:gdo_margin].round(1).to_s.rjust(12)}"
      end
    end

    # Parlay Rankings
    puts "\n" + "=" * 70
    puts "🎰 PARLAY RANKINGS (#{today})"
    puts "=" * 70

    rankings = analyzer.parlay_rankings(today)

    if rankings.empty?
      puts "No sweet spot games for parlay."
    else
      puts "#{'Rank'.ljust(6)} #{'Tier'.ljust(8)} #{'Game'.ljust(20)} #{'Pick'.ljust(8)} #{'Spread'.rjust(8)} #{'GDO Margin'.rjust(12)}"
      puts "-" * 70
      
      rankings.each_with_index do |a, i|
        game = a[:game]
        matchup = "#{game.away_team.code} @ #{game.home_team.code}"
        tier_label = a[:tier] == 2.0 ? "SS2.0" : "SS1.0"
        
        puts "#{(i + 1).to_s.ljust(6)} #{tier_label.ljust(8)} #{matchup.ljust(20)} #{a[:gdo_pick].code.ljust(8)} #{a[:spread_size].to_s.rjust(8)} #{a[:gdo_margin].round(1).to_s.rjust(12)}"
      end
    end

    puts "\n" + "=" * 70
  end
end