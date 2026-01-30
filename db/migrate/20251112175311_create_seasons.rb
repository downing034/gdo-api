class CreateSeasons < ActiveRecord::Migration[8.0]
  def change
    create_table :seasons, id: :uuid do |t|
      t.references :league, null: false, foreign_key: true, type: :uuid
      t.string :name, null: false
      t.date :start_date, null: false
      t.date :end_date, null: false
      t.boolean :active, default: false, null: false

      t.timestamps
    end

    add_index :seasons, [:league_id, :name], unique: true
    add_index :seasons, :active
  end
end