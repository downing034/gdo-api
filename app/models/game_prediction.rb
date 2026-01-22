class GamePrediction < ApplicationRecord
  belongs_to :game
  belongs_to :data_source
  belongs_to :predicted_winner, class_name: 'Team', optional: true

  validates :model_version, presence: true
  validates :generated_at, presence: true

  # Scopes
  scope :score_predictions, -> { where.not(away_predicted_score: nil, home_predicted_score: nil) }
  scope :winner_predictions, -> { where.not(predicted_winner_id: nil) }
  scope :for_model, ->(version) { where(model_version: version) }

  def score_prediction?
    away_predicted_score.present? && home_predicted_score.present?
  end

  def winner_prediction?
    predicted_winner_id.present?
  end
end