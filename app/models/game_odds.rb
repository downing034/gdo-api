class GameOdds < ApplicationRecord
  belongs_to :game
  belongs_to :spread_favorite_team, class_name: 'Team', optional: true
  belongs_to :moneyline_favorite_team, class_name: 'Team', optional: true
  belongs_to :data_source, optional: true

  # Validations
  validates :spread_value, numericality: { less_than: 0 }, allow_nil: true
  validates :total_line, numericality: { greater_than: 0 }, allow_nil: true
  validate :spread_fields_together

  # Scopes
  scope :current, -> { order(fetched_at: :desc).limit(1) }
  scope :for_game, ->(game) { where(game: game) }

  private

  def spread_fields_together
    if spread_favorite_team_id.present? ^ spread_value.present?
      errors.add(:base, "Spread favorite team and spread value must both be present or both be absent")
    end
  end
end