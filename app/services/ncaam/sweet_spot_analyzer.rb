module Ncaam
  class SweetSpotAnalyzer
    SWEET_SPOT_2_MIN_SPREAD = 13
    SWEET_SPOT_1_MIN_SPREAD = 10
    MIN_GDO_MARGIN = 8

    def initialize(league_code: 'ncaam')
      @league = League.find_by!(code: league_code)
    end

    def analyze_game(game)
      gdo_pred = game.game_predictions.find { |p| p.model_version == 'v1' }
      sl_pred = game.game_predictions.find { |p| p.model_version == 'sl' }
      odds = game.game_odds.where(is_opening: false).order(fetched_at: :desc).first

      return nil unless gdo_pred && sl_pred && odds && odds.spread_value.present?

      gdo_away = gdo_pred.away_predicted_score.to_f
      gdo_home = gdo_pred.home_predicted_score.to_f
      sl_away = sl_pred.away_predicted_score.to_f
      sl_home = sl_pred.home_predicted_score.to_f

      gdo_pick = gdo_away > gdo_home ? game.away_team : game.home_team
      sl_pick = sl_away > sl_home ? game.away_team : game.home_team
      models_agree = gdo_pick == sl_pick

      gdo_margin = (gdo_away - gdo_home).abs
      spread_size = odds.spread_value.abs
      spread_favorite = odds.spread_favorite_team
      gdo_picks_favorite = gdo_pick == spread_favorite

      # Determine if GDO was correct (only for final games)
      gdo_correct = nil
      if game.final? && game.game_result&.final?
        away_score = game.game_result.away_score
        home_score = game.game_result.home_score
        actual_winner = away_score > home_score ? game.away_team : game.home_team
        gdo_correct = gdo_pick == actual_winner
      end

      # Determine sweet spot tier
      tier = nil
      if models_agree && gdo_picks_favorite && gdo_margin >= MIN_GDO_MARGIN
        if spread_size >= SWEET_SPOT_2_MIN_SPREAD
          tier = 2.0
        elsif spread_size >= SWEET_SPOT_1_MIN_SPREAD
          tier = 1.0
        end
      end

      {
        game: game,
        gdo_pick: gdo_pick,
        sl_pick: sl_pick,
        models_agree: models_agree,
        gdo_margin: gdo_margin,
        spread_size: spread_size,
        spread_favorite: spread_favorite,
        gdo_picks_favorite: gdo_picks_favorite,
        gdo_correct: gdo_correct,
        tier: tier,
        moneyline_favorite_odds: odds.moneyline_favorite_odds,
        moneyline_underdog_odds: odds.moneyline_underdog_odds
      }
    end

    def games_for_date(date)
      Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
          .where(league: @league, game_date: date)
          .order(:start_time)
    end

    def sweet_spot_games(date)
      games_for_date(date)
        .map { |g| analyze_game(g) }
        .compact
        .select { |a| a[:tier].present? }
    end

    def accuracy_stats(start_date:, end_date:)
      games = Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
                  .where(league: @league, game_date: start_date..end_date)
                  .where(status: 'final')

      analyses = games.map { |g| analyze_game(g) }.compact

      thresholds = [10, 12, 13, 14, 15]
      
      stats = thresholds.map do |threshold|
        qualifying = analyses.select do |a|
          a[:models_agree] && 
          a[:gdo_picks_favorite] && 
          a[:gdo_margin] >= MIN_GDO_MARGIN && 
          a[:spread_size] >= threshold
        end

        correct = qualifying.count { |a| a[:gdo_correct] }
        total = qualifying.count

        {
          threshold: threshold,
          correct: correct,
          total: total,
          accuracy: total > 0 ? (correct.to_f / total * 100).round(1) : 0
        }
      end

      stats
    end

    def parlay_rankings(date)
      sweet_spot_games(date)
        .sort_by { |a| [a[:tier] == 2.0 ? 0 : 1, -a[:spread_size]] }
    end
  end
end