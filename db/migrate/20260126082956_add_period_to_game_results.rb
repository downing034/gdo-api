class AddPeriodToGameResults < ActiveRecord::Migration[8.0]
  def change
    add_column :game_results, :period, :integer
  end
end