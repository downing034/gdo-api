class Game < ApplicationRecord
  belongs_to :season
  belongs_to :league
  belongs_to :home_team, class_name: 'Team'
  belongs_to :away_team, class_name: 'Team'
  
  has_one :game_result, dependent: :destroy
  has_many :game_odds, dependent: :destroy
  has_many :game_predictions, dependent: :destroy
  has_many :basketball_game_team_stats, dependent: :destroy
  has_many :basketball_game_player_stats, dependent: :destroy

  enum :status, {
    scheduled: "scheduled",
    in_progress: "in_progress",
    final: "final",
    postponed: "postponed",
    delayed: "delayed",
    cancelled: "cancelled"
  }

  validates :start_time, presence: true
  validate :teams_must_be_different
  validate :teams_must_be_in_same_league
  validates :external_id, uniqueness: { scope: :league_id }, allow_nil: true

  scope :for_date, ->(date) { 
    start_of_day = date.in_time_zone('America/Denver').beginning_of_day.utc
    end_of_day = date.in_time_zone('America/Denver').end_of_day.utc
    where(start_time: start_of_day..end_of_day)
  }
  scope :for_league, ->(league) { where(league: league) }
  scope :for_team, ->(team) { where('home_team_id = ? OR away_team_id = ?', team.id, team.id) }
  scope :completed, -> { where(status: :final) }
  scope :upcoming, -> { where(status: [:scheduled, :delayed]) }
  scope :with_results, -> { joins(:game_result).where(game_results: { final: true }) }
  scope :recent_first, -> { order(start_time: :desc) }
  scope :chronological, -> { order(start_time: :asc) }
  scope :active, -> { where(is_stale: false) }
  scope :stale, -> { where(is_stale: true) }
  scope :playable, -> { where.not(status: [:postponed, :cancelled]) }

  # Team-specific scopes
  scope :last_n_for_team, ->(team, n) { for_team(team).with_results.recent_first.limit(n) }
  scope :next_for_team, ->(team) { for_team(team).upcoming.chronological.first }
  scope :current_season, ->(year = Date.current.year) { 
    where('EXTRACT(year from start_time) = ?', year) 
  }

  def game_date
    start_time&.in_time_zone('America/Denver')&.to_date
  end

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
    
    unless home_team.leagues.include?(league)
      errors.add(:home_team, "must belong to the #{league.code} league")
    end
    
    unless away_team.leagues.include?(league)
      errors.add(:away_team, "must belong to the #{league.code} league")
    end
  end
end