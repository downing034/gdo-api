class Sport < ApplicationRecord
  has_many :leagues, dependent: :restrict_with_exception

  validates :code, presence: true, uniqueness: true
  validates :name, presence: true
end
