class CreateGames < ActiveRecord::Migration[8.0]
  def change
    create_table :games, id: :uuid do |t|
      t.references :league, null: false, foreign_key: true, type: :uuid
      t.date :game_date, null: false
      t.references :home_team, null: false, foreign_key: { to_table: :teams }, type: :uuid
      t.references :away_team, null: false, foreign_key: { to_table: :teams }, type: :uuid
      t.datetime :start_time
      t.integer :status, default: 0

      t.timestamps
    end

    # Ensure no duplicate games (including start_time for doubleheaders)
    add_index :games, [:league_id, :game_date, :home_team_id, :away_team_id, :start_time], 
              unique: true, name: 'index_games_on_unique_game'
    
    # Common query patterns (references already creates individual indexes)
    add_index :games, [:league_id, :game_date]

    # Validation at DB level
    add_check_constraint :games, "home_team_id != away_team_id", name: "games_different_teams"
  end
end