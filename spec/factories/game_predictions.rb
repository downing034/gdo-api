# frozen_string_literal: true

FactoryBot.define do
  factory :game_prediction do
    game
    data_source
    model_version { 'v1' }
    home_predicted_score { 75.5 }
    away_predicted_score { 70.2 }
    predicted_winner { game.home_team }
    confidence { 0.65 }
    generated_at { Time.current }
  end
end