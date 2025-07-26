class CreateTeams < ActiveRecord::Migration[8.0]
  def change
    create_table :teams, id: :uuid do |t|
      t.string :code
      t.string :location_name
      t.string :nickname
      t.boolean :active
      t.references :league, null: false, foreign_key: true, type: :uuid

      t.timestamps
    end
  end
end
