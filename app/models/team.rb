class Team < ApplicationRecord
  has_many :leagues_teams
  has_many :leagues, through: :leagues_teams
  belongs_to :venue, optional: true
  has_many :team_identifiers, dependent: :destroy

  validates :code, presence: true, uniqueness: true
  validates :nickname, presence: true
  validates :active, inclusion: { in: [true, false] }

  def display_name
    "#{location_name} #{nickname}"
  end
end