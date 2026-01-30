class RemoveGameDateFromGames < ActiveRecord::Migration[8.0]
  def change
    remove_column :games, :game_date, :date
  end
end