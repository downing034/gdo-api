class CreateBasketballGameTeamStats < ActiveRecord::Migration[8.0]
  def change
    create_table :basketball_game_team_stats, id: :uuid do |t|
      t.references :game, type: :uuid, null: false, foreign_key: true
      t.references :team, type: :uuid, null: false, foreign_key: true

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

      t.integer :points_off_turnovers
      t.integer :fast_break_points
      t.integer :points_in_paint

      t.integer :largest_lead
      t.integer :lead_changes
      t.decimal :time_leading_pct, precision: 5, scale: 2

      t.timestamps
    end

    add_index :basketball_game_team_stats, [:game_id, :team_id], unique: true
  end
end