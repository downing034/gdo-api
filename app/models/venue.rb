class Venue < ApplicationRecord
  has_many :teams

  validates :name, presence: true, uniqueness: true
  validates :latitude, numericality: true, allow_nil: true
  validates :longitude, numericality: true, allow_nil: true
  validates :is_active, inclusion: { in: [true, false] }
end