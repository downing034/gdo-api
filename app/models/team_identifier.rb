class TeamIdentifier < ApplicationRecord
  belongs_to :team
  belongs_to :league
  belongs_to :data_source

  validates :external_code, presence: true
  validates :external_code, uniqueness: {
    scope: [:data_source_id, :league_id],
    message: "must be unique per data source and league"
  }

  scope :active, -> { where(active: true) }
end