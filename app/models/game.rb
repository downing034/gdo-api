class Game < ApplicationRecord
  belongs_to :league
  belongs_to :home_team, class_name: 'Team'
  belongs_to :away_team, class_name: 'Team'
  
  has_one :game_result, dependent: :destroy

  enum status: {
    scheduled: 0,
    in_progress: 1,
    final: 2,
    postponed: 3,
    delayed: 4,
    cancelled: 5
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
    return unless home_team && away_team
    
    if home_team.league_id != away_team.league_id
      errors.add(:base, "Teams must be in the same league")
    end

    if league && (home_team.league_id != league.id || away_team.league_id != league.id)
      errors.add(:base, "Teams must belong to the specified league")
    end
  end
end