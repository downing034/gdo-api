class AddIsOpeningToGameOdds < ActiveRecord::Migration[8.0]
  def change
    add_column :game_odds, :is_opening, :boolean, default: false, null: false
  end
end
