=begin
In the the terminal, this script can be used to export a single days game data
which will match the #{LEAGUE}_model_tracking.csv format for all but MLB

$ rails games:export_ncaam[2026-01-20,true] | pbcopy

pass in true if you want the header, false if you want to exclude it

change the league and data source for the odds if desired
change the gdo_pred if different pred are desired
=end

namespace :games do
  desc "Export games to CSV for predictions (default: today, override with date like ['2026-01-20'])"
  task :export_ncaam, [:date, :include_header] => :environment do |t, args|
    include_header = args[:include_header] != 'false'

    date = args[:date] ? Date.parse(args[:date]) : Date.current
    league = League.find_by!(code: 'ncaam')
    espn_source = DataSource.find_by!(code: 'espn')
    
    games = Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
                .where(league: league, game_date: date)
                .order(:start_time)
    
    if include_header
      puts "id,date,start_time,away_team,home_team,sl_away_pred,sl_home_pred,gdo_away_pred,gdo_home_pred,away_result,home_result,moneyline_favorite_team,total_line,runline_favorite_team,runline_value,underdog_runline_odds,favorite_runline_odds,over_total_line_odds,under_total_line_odds,underdog_moneyline_odds,favorite_moneyline_odds"
    end
    
    games.each_with_index do |game, index|
      id = index + 1
      game_date = game.game_date.strftime('%Y%m%d')
      start_time = game.start_time.in_time_zone('America/Denver').strftime('%H:%M')
      away_team = game.away_team.code
      home_team = game.home_team.code
      
      sl_pred = game.game_predictions.find { |p| p.model_version == 'sl' }
      gdo_pred = game.game_predictions.find { |p| p.model_version == 'v1' }
      
      sl_away = sl_pred&.away_predicted_score&.to_i
      sl_home = sl_pred&.home_predicted_score&.to_i
      gdo_away = gdo_pred&.away_predicted_score&.to_i
      gdo_home = gdo_pred&.home_predicted_score&.to_i
      
      away_result = game.game_result&.final? ? game.game_result.away_score : nil
      home_result = game.game_result&.final? ? game.game_result.home_score : nil
      
      odds = game.game_odds
                 .where(data_source: espn_source)
                 .where(is_opening: false)
                 .order(fetched_at: :desc)
                 .first
      
      ml_fav_team = odds&.moneyline_favorite_team&.code
      total_line = odds&.total_line&.to_f
      spread_fav_team = odds&.spread_favorite_team&.code
      spread_value = odds&.spread_value&.to_f
      underdog_spread_odds = odds&.spread_underdog_odds
      favorite_spread_odds = odds&.spread_favorite_odds
      over_odds = odds&.over_odds
      under_odds = odds&.under_odds
      underdog_ml_odds = odds&.moneyline_underdog_odds
      favorite_ml_odds = odds&.moneyline_favorite_odds
      
      puts [
        id, game_date, start_time, away_team, home_team,
        sl_away, sl_home, gdo_away, gdo_home,
        away_result, home_result,
        ml_fav_team, total_line, spread_fav_team, spread_value,
        underdog_spread_odds, favorite_spread_odds,
        over_odds, under_odds,
        underdog_ml_odds, favorite_ml_odds
      ].join(',')
    end
  end
end