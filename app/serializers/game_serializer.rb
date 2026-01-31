# frozen_string_literal: true

class GameSerializer < BaseSerializer
  def as_json
    data = {
      id: object.id,
      league: object.league.code,
      season: object.season.name,
      start_time: object.start_time.iso8601,
      status: object.status,
      home_team: TeamSerializer.new(object.home_team).as_json,
      away_team: TeamSerializer.new(object.away_team).as_json
    }

    data[:result] = result_json if include_result?
    data[:odds] = odds_json if include_odds?
    data[:predictions] = predictions_json if include_predictions?

    data
  end

  private

  def include_result?
    options[:include_result] && object.game_result.present?
  end

  def include_odds?
    options[:include_odds]
  end

  def include_predictions?
    options[:include_predictions]
  end

  def result_json
    return nil unless object.game_result

    {
      home_score: object.game_result.home_score,
      away_score: object.game_result.away_score,
      final: object.game_result.final,
      period_scores: object.game_result.period_scores
    }
  end

  def odds_json
    odds = object.game_odds.where(is_opening: false).order(fetched_at: :desc).first
    return nil unless odds

    GameOddsSerializer.new(odds).as_json
  end

  def predictions_json
    object.game_predictions.includes(:data_source).each_with_object({}) do |pred, hash|
      hash[pred.data_source.code] = GamePredictionSerializer.new(pred).as_json
    end
  end
end