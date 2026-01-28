class CreateBasketballGamePlayerStats < ActiveRecord::Migration[8.0]
  def change
    create_table :basketball_game_player_stats, id: :uuid do |t|
      t.references :game, type: :uuid, null: false, foreign_key: true
      t.references :player, type: :uuid, null: false, foreign_key: true
      t.references :team, type: :uuid, null: false, foreign_key: true

      t.integer :minutes_played

      t.integer :points
      t.integer :field_goals_made
      t.integer :field_goals_attempted
      t.integer :three_pointers_made
      t.integer :three_pointers_attempted
      t.integer :free_throws_made
      t.integer :free_throws_attempted

      t.integer :offensive_rebounds
      t.integer :defensive_rebounds

      t.integer :assists
      t.integer :steals
      t.integer :blocks
      t.integer :turnovers
      t.integer :fouls

      t.timestamps
    end

    add_index :basketball_game_player_stats, [:game_id, :player_id], unique: true
    add_index :basketball_game_player_stats, [:player_id, :game_id], name: 'idx_bb_player_stats_player_game'
  end
end