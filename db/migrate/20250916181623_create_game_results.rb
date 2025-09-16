class CreateGameResults < ActiveRecord::Migration[8.0]
  def change
    create_table :game_results, id: :uuid do |t|
      t.references :game, null: false, foreign_key: true, type: :uuid, index: { unique: true }
      t.integer :home_score
      t.integer :away_score
      t.boolean :final, default: false
      t.json :period_scores
      t.text :notes

      t.timestamps
    end
  end
end