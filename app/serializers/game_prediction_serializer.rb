# frozen_string_literal: true

class GamePredictionSerializer < BaseSerializer
  def as_json
    {
      home_score: object.home_predicted_score&.to_f,
      away_score: object.away_predicted_score&.to_f,
      predicted_winner: object.predicted_winner&.code,
      confidence: object.confidence&.to_f,
      model_version: object.model_version,
      generated_at: object.generated_at&.iso8601
    }
  end
end