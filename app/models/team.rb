class Team < ApplicationRecord
  belongs_to :league
  belongs_to :venue, optional: true

  validates :code, presence: true, uniqueness: { scope: :league_id }
  validates :location_name, presence: true
  validates :nickname, presence: true
  validates :active, inclusion: { in: [true, false] }

  def display_name
    "#{location_name} #{nickname}"
  end
end
