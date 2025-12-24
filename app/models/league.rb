class League < ApplicationRecord
  belongs_to :sport
  has_many :teams, dependent: :restrict_with_exception
  has_many :team_identifiers, dependent: :destroy
  has_many :seasons, dependent: :restrict_with_exception

  validates :code, presence: true, uniqueness: { scope: :sport_id }
  validates :name, presence: true
  validates :active, inclusion: { in: [true, false] }
end