class Season < ApplicationRecord
  belongs_to :league
  has_many :games, dependent: :restrict_with_exception
end