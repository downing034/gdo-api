class CreateGameOdds < ActiveRecord::Migration[8.0]
  def change
    create_table :game_odds, id: :uuid do |t|
      t.references :game, null: false, foreign_key: true, type: :uuid
      t.references :spread_favorite_team, null: true, foreign_key: { to_table: :teams }, type: :uuid
      t.decimal :spread_value
      t.integer :spread_favorite_odds
      t.integer :spread_underdog_odds
      t.decimal :total_line
      t.integer :over_odds
      t.integer :under_odds
      t.references :moneyline_favorite_team, null: true, foreign_key: { to_table: :teams }, type: :uuid
      t.integer :moneyline_favorite_odds
      t.integer :moneyline_underdog_odds
      t.references :data_source, null: false, foreign_key: true, type: :uuid
      t.datetime :fetched_at

      t.timestamps
    end

    add_index :game_odds, [:game_id, :fetched_at]
  end
end
