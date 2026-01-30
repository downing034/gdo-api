class CreatePlayers < ActiveRecord::Migration[8.0]
  def change
    create_table :players, id: :uuid do |t|
      t.string :external_id, null: false
      t.references :data_source, type: :uuid, null: false, foreign_key: true
      t.references :team, type: :uuid, foreign_key: true
      t.string :name, null: false
      t.string :position
      t.boolean :active, default: true, null: false

      t.timestamps
    end

    add_index :players, [:data_source_id, :external_id], unique: true
  end
end