class DataSource < ApplicationRecord
  has_many :team_identifiers, dependent: :destroy

  validates :code, presence: true, uniqueness: true
  validates :name, presence: true
end