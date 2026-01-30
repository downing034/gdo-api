class AddExternalTrackingToGames < ActiveRecord::Migration[8.0]
  def change
    add_column :games, :external_id, :string
    add_column :games, :is_stale, :boolean, default: false
    
    add_index :games, [:league_id, :external_id], unique: true
  end
end