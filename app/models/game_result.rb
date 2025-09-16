class GameResult < ApplicationRecord
  belongs_to :game

  validates :home_score, :away_score, presence: true, if: :final?
  validates :home_score, :away_score, numericality: { greater_than_or_equal_to: 0 }, allow_nil: true

  def winner
    return nil if tie? || !final?
    home_score > away_score ? game.home_team : game.away_team
  end

  def loser
    return nil if tie? || !final?
    home_score < away_score ? game.home_team : game.away_team  
  end

  def tie?
    return false unless final? && home_score && away_score
    home_score == away_score
  end

  def home_won?
    final? && home_score && away_score && home_score > away_score
  end

  def away_won?
    final? && home_score && away_score && away_score > home_score
  end

  def final_score
    return nil unless home_score && away_score
    "#{away_score}-#{home_score}"
  end

  def score_differential
    return nil unless home_score && away_score
    (home_score - away_score).abs
  end
end