class Game < ApplicationRecord
  belongs_to :league
  belongs_to :home_team, class_name: 'Team'
  belongs_to :away_team, class_name: 'Team'

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
  scope :completed, -> { where(status: :final) }
  scope :upcoming, -> { where(status: [:scheduled, :delayed]) }

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