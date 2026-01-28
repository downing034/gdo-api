class BasketballGameTeamStat < ApplicationRecord
  belongs_to :game
  belongs_to :team

  validates :game_id, uniqueness: { scope: :team_id }

  # Computed percentages
  def field_goal_pct
    return nil unless field_goals_attempted&.positive?
    field_goals_made.to_f / field_goals_attempted
  end

  def three_point_pct
    return nil unless three_pointers_attempted&.positive?
    three_pointers_made.to_f / three_pointers_attempted
  end

  def free_throw_pct
    return nil unless free_throws_attempted&.positive?
    free_throws_made.to_f / free_throws_attempted
  end

  def total_rebounds
    (offensive_rebounds || 0) + (defensive_rebounds || 0)
  end
end