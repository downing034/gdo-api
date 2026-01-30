class CreateLeagues < ActiveRecord::Migration[8.0]
  def change
    create_table :leagues, id: :uuid do |t|
      t.references :sport, null: false, foreign_key: true, type: :uuid
      t.string :code
      t.string :name
      t.string :display_name
      t.boolean :has_conferences
      t.boolean :active

      t.timestamps
    end
  end
end
