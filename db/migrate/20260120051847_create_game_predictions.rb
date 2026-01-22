class CreateGamePredictions < ActiveRecord::Migration[8.0]
  def change
    create_table :game_predictions, id: :uuid do |t|
      t.references :game, null: false, foreign_key: true, type: :uuid
      t.string :model_version, null: false
      t.references :data_source, null: false, foreign_key: true, type: :uuid
      t.decimal :away_predicted_score
      t.decimal :home_predicted_score
      t.references :predicted_winner, null: true, foreign_key: { to_table: :teams }, type: :uuid
      t.decimal :confidence
      t.datetime :generated_at

      t.timestamps
    end

    # Unique constraint: one prediction per game/model/source/time
    add_index :game_predictions, [:game_id, :model_version, :data_source_id, :generated_at], 
              unique: true, name: 'index_game_predictions_unique'
    
    # Query indexes
    add_index :game_predictions, [:game_id, :model_version]
  end
end