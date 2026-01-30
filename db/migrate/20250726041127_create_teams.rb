class CreateTeams < ActiveRecord::Migration[8.0]
  def change
    create_table :teams, id: :uuid do |t|
      t.string :code
      t.string :location_name
      t.string :nickname
      t.boolean :active

      t.timestamps
    end
  end
end
