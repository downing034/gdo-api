class Player < ApplicationRecord
  belongs_to :data_source
  belongs_to :team, optional: true

  has_many :basketball_game_player_stats, dependent: :destroy

  validates :external_id, presence: true, uniqueness: { scope: :data_source_id }
  validates :name, presence: true
end