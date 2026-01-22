class ChangeGamesStatusToString < ActiveRecord::Migration[8.0]
  def change
    change_column :games, :status, :string, default: "scheduled"
  end
end
