module Ncaam
  class SweetSpotAnalyzer
    # Original Sweet Spot thresholds (V3 point prediction based)
    # These remain unchanged
    SWEET_SPOT_2_MIN_SPREAD = 13
    SWEET_SPOT_1_MIN_SPREAD = 10
    MIN_GDO_MARGIN = 8

    # V4 Sweet Spot thresholds (data-driven from analysis of 1,581 games)
    # 
    # Analysis findings:
    # - V4 conf >=80% + spread >=13 + fav: 98.5% (134/136)
    # - V4 conf >=80% + spread >=15 + fav: 98.9% (93/94)
    # - V4 conf >=70% + spread >=13 + fav: 97.4% (147/151)
    # - V4 conf >=75% + spread >=10 + fav: 95.6% (238/249)
    # - V4 conf >=60% + underdog: 81.9% (118/144)
    #
    # V4 Sweet Spot 2.0: Highest accuracy (98%+)
    V4_SS2_MIN_CONFIDENCE = 80
    V4_SS2_MIN_SPREAD = 13
    
    # V4 Sweet Spot 1.0: Very good accuracy (97%+)
    V4_SS1_MIN_CONFIDENCE = 70
    V4_SS1_MIN_SPREAD = 13
    
    # V4 Underdog Sweet Spot: V4 picks underdog with conf >=60% (82% accuracy)
    V4_UNDERDOG_MIN_CONFIDENCE = 60

    def initialize(league_code: 'ncaam')
      @league = League.find_by!(code: league_code)
    end

    def analyze_game(game)
      gdo_pred = game.game_predictions.find { |p| p.model_version.start_with?('v3') }
      sl_pred = game.game_predictions.find { |p| p.model_version == 'sl' }
      v4_pred = game.game_predictions.find { |p| p.model_version.start_with?('v4') }
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

      # V4 data
      v4_pick = v4_pred&.predicted_winner
      v4_confidence = v4_pred&.confidence&.to_f
      v4_picks_favorite = v4_pick == spread_favorite if v4_pick && spread_favorite
      v4_picks_underdog = v4_pick && spread_favorite && v4_pick != spread_favorite
      v4_agrees_with_gdo = v4_pick == gdo_pick if v4_pick
      all_models_agree = v4_pick == gdo_pick && gdo_pick == sl_pick if v4_pick

      # Determine if predictions were correct (only for final games)
      gdo_correct = nil
      v4_correct = nil
      if game.final? && game.game_result&.final?
        away_score = game.game_result.away_score
        home_score = game.game_result.home_score
        actual_winner = away_score > home_score ? game.away_team : game.home_team
        gdo_correct = gdo_pick == actual_winner
        v4_correct = v4_pick == actual_winner if v4_pick
      end

      # Determine original sweet spot tier (unchanged)
      tier = nil
      if models_agree && gdo_picks_favorite && gdo_margin >= MIN_GDO_MARGIN
        if spread_size >= SWEET_SPOT_2_MIN_SPREAD
          tier = 2.0
        elsif spread_size >= SWEET_SPOT_1_MIN_SPREAD
          tier = 1.0
        end
      end

      # Determine V4 sweet spot tier (data-driven from 1,581 games)
      v4_tier = nil
      if v4_pick && v4_confidence
        if v4_picks_favorite
          # V4 Sweet Spot 2.0: conf >=80% + spread >=13 (98.5% historical)
          if v4_confidence >= V4_SS2_MIN_CONFIDENCE && spread_size >= V4_SS2_MIN_SPREAD
            v4_tier = 2.0
          # V4 Sweet Spot 1.0: conf >=70% + spread >=13 (97.4% historical)
          elsif v4_confidence >= V4_SS1_MIN_CONFIDENCE && spread_size >= V4_SS1_MIN_SPREAD
            v4_tier = 1.0
          end
        elsif v4_picks_underdog
          # V4 Underdog: conf >=60% picking underdog (81.9% historical)
          if v4_confidence >= V4_UNDERDOG_MIN_CONFIDENCE
            v4_tier = :underdog
          end
        end
      end

      {
        game: game,
        # Original sweet spot fields
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
        moneyline_underdog_odds: odds.moneyline_underdog_odds,
        # V4 sweet spot fields
        v4_pick: v4_pick,
        v4_confidence: v4_confidence,
        v4_picks_favorite: v4_picks_favorite,
        v4_picks_underdog: v4_picks_underdog,
        v4_agrees_with_gdo: v4_agrees_with_gdo,
        all_models_agree: all_models_agree,
        v4_correct: v4_correct,
        v4_tier: v4_tier
      }
    end

    def games_for_date(date)
      Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
          .where(league: @league)
          .for_date(date)
          .order(:start_time)
    end

    # =========================================
    # ORIGINAL SWEET SPOT METHODS (unchanged)
    # =========================================

    def sweet_spot_games(date)
      games_for_date(date)
        .map { |g| analyze_game(g) }
        .compact
        .select { |a| a[:tier].present? }
    end

    def accuracy_stats(start_date:, end_date:)
      start_time_begin = start_date.in_time_zone('America/Denver').beginning_of_day.utc
      start_time_end = end_date.in_time_zone('America/Denver').end_of_day.utc

      games = Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
                  .where(league: @league, start_time: start_time_begin..start_time_end)
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

    # =========================================
    # V4 SWEET SPOT METHODS (data-driven)
    # =========================================

    # All V4 sweet spot games (favorites only, tier 1.0 or 2.0)
    def v4_sweet_spot_games(date)
      games_for_date(date)
        .map { |g| analyze_game(g) }
        .compact
        .select { |a| a[:v4_tier].is_a?(Numeric) }
    end

    # V4 underdog picks (conf >=60%, picking against spread favorite)
    def v4_underdog_picks(date)
      games_for_date(date)
        .map { |g| analyze_game(g) }
        .compact
        .select { |a| a[:v4_tier] == :underdog }
    end

    # Games that qualify for BOTH original SS and V4 SS
    def combined_sweet_spot_games(date)
      games_for_date(date)
        .map { |g| analyze_game(g) }
        .compact
        .select { |a| a[:tier].present? && a[:v4_tier].is_a?(Numeric) }
    end

    # V4 accuracy stats by confidence and spread thresholds
    def v4_accuracy_stats(start_date:, end_date:)
      start_time_begin = start_date.in_time_zone('America/Denver').beginning_of_day.utc
      start_time_end = end_date.in_time_zone('America/Denver').end_of_day.utc

      games = Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
                  .where(league: @league, start_time: start_time_begin..start_time_end)
                  .where(status: 'final')

      analyses = games.map { |g| analyze_game(g) }.compact.select { |a| a[:v4_pick].present? }

      # Test combinations that emerged from data analysis
      [
        { name: 'V4 SS 2.0 (conf>=70, spread>=13, fav)', conf: 70, spread: 13, fav_only: true },
        { name: 'V4 SS 1.0 (conf>=70, spread>=10, fav)', conf: 70, spread: 10, fav_only: true },
        { name: 'V4 conf>=75 + fav', conf: 75, spread: 0, fav_only: true },
        { name: 'V4 conf>=80 + fav', conf: 80, spread: 0, fav_only: true },
        { name: 'V4 underdog (conf>=60)', conf: 60, spread: 0, fav_only: false, underdog_only: true },
      ].map do |criteria|
        qualifying = analyses.select do |a|
          next false unless a[:v4_confidence] && a[:v4_confidence] >= criteria[:conf]
          next false if criteria[:spread] > 0 && a[:spread_size] < criteria[:spread]
          next false if criteria[:fav_only] && !a[:v4_picks_favorite]
          next false if criteria[:underdog_only] && !a[:v4_picks_underdog]
          true
        end

        correct = qualifying.count { |a| a[:v4_correct] }
        total = qualifying.count

        {
          name: criteria[:name],
          correct: correct,
          total: total,
          accuracy: total > 0 ? (correct.to_f / total * 100).round(1) : 0
        }
      end
    end

    # Compare original SS to SS + V4 agreement
    def combined_accuracy_stats(start_date:, end_date:)
      start_time_begin = start_date.in_time_zone('America/Denver').beginning_of_day.utc
      start_time_end = end_date.in_time_zone('America/Denver').end_of_day.utc

      games = Game.includes(:home_team, :away_team, :game_result, :game_odds, :game_predictions)
                  .where(league: @league, start_time: start_time_begin..start_time_end)
                  .where(status: 'final')

      analyses = games.map { |g| analyze_game(g) }.compact

      # Original SS 2.0
      ss2_games = analyses.select { |a| a[:tier] == 2.0 }
      ss2_correct = ss2_games.count { |a| a[:gdo_correct] }
      ss2_total = ss2_games.count

      # SS 2.0 where V4 also agrees
      ss2_v4_agree = ss2_games.select { |a| a[:v4_agrees_with_gdo] }
      ss2_v4_correct = ss2_v4_agree.count { |a| a[:gdo_correct] }
      ss2_v4_total = ss2_v4_agree.count

      # V4 SS 2.0 standalone
      v4_ss2 = analyses.select do |a| 
        a[:v4_confidence] && a[:v4_confidence] >= V4_SS2_MIN_CONFIDENCE &&
        a[:spread_size] >= V4_SS2_MIN_SPREAD &&
        a[:v4_picks_favorite]
      end
      v4_ss2_correct = v4_ss2.count { |a| a[:v4_correct] }
      v4_ss2_total = v4_ss2.count

      {
        original_ss2: {
          correct: ss2_correct,
          total: ss2_total,
          accuracy: ss2_total > 0 ? (ss2_correct.to_f / ss2_total * 100).round(1) : 0
        },
        ss2_plus_v4_agree: {
          correct: ss2_v4_correct,
          total: ss2_v4_total,
          accuracy: ss2_v4_total > 0 ? (ss2_v4_correct.to_f / ss2_v4_total * 100).round(1) : 0
        },
        v4_ss2_standalone: {
          correct: v4_ss2_correct,
          total: v4_ss2_total,
          accuracy: v4_ss2_total > 0 ? (v4_ss2_correct.to_f / v4_ss2_total * 100).round(1) : 0
        }
      }
    end

    # V4 parlay rankings (by tier then confidence)
    def v4_parlay_rankings(date)
      v4_sweet_spot_games(date)
        .sort_by { |a| [a[:v4_tier] == 2.0 ? 0 : 1, -(a[:v4_confidence] || 0)] }
    end

    # Combined parlay: games in both original SS and V4 SS
    def combined_parlay_rankings(date)
      combined_sweet_spot_games(date)
        .sort_by { |a| [a[:tier] == 2.0 ? 0 : 1, a[:v4_tier] == 2.0 ? 0 : 1, -a[:spread_size]] }
    end
  end
end