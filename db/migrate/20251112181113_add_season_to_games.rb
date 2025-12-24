class AddSeasonToGames < ActiveRecord::Migration[8.0]
  def change
    add_reference :games, :season, null: false, foreign_key: true, type: :uuid
  end
end