class Game < ApplicationRecord
  belongs_to :season
  belongs_to :league
  belongs_to :home_team, class_name: 'Team'
  belongs_to :away_team, class_name: 'Team'
  
  has_one :game_result, dependent: :destroy
  has_many :game_odds, dependent: :destroy
  has_many :game_predictions, dependent: :destroy

  enum :status, {
    scheduled: "scheduled",
    in_progress: "in_progress",
    final: "final",
    postponed: "postponed",
    delayed: "delayed",
    cancelled: "cancelled"
  }

  validates :game_date, presence: true
  validate :teams_must_be_different
  validate :teams_must_be_in_same_league

  scope :for_date, ->(date) { where(game_date: date) }
  scope :for_league, ->(league) { where(league: league) }
  scope :for_team, ->(team) { where('home_team_id = ? OR away_team_id = ?', team.id, team.id) }
  scope :completed, -> { where(status: :final) }
  scope :upcoming, -> { where(status: [:scheduled, :delayed]) }
  scope :with_results, -> { joins(:game_result).where(game_results: { final: true }) }
  scope :recent_first, -> { order(game_date: :desc, start_time: :desc) }
  scope :chronological, -> { order(game_date: :asc, start_time: :asc) }
  
  # Team-specific scopes
  scope :last_n_for_team, ->(team, n) { for_team(team).with_results.recent_first.limit(n) }
  scope :next_for_team, ->(team) { for_team(team).upcoming.chronological.first }
  scope :current_season, ->(year = Date.current.year) { where('EXTRACT(year from game_date) = ?', year) }

  def completed?
    final?
  end

  def teams
    [home_team, away_team]
  end

  private

  def teams_must_be_different
    errors.add(:away_team, "can't be the same as home team") if home_team_id == away_team_id
  end

  def teams_must_be_in_same_league
  return unless home_team && away_team && league
  
  # Check that both teams belong to the game's league
  unless home_team.leagues.include?(league)
    errors.add(:home_team, "must belong to the #{league.code} league")
  end
  
  unless away_team.leagues.include?(league)
    errors.add(:away_team, "must belong to the #{league.code} league")
  end
end
end