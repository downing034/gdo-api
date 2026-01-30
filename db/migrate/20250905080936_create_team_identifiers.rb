class CreateTeamIdentifiers < ActiveRecord::Migration[8.0]
  def change
    create_table :team_identifiers, id: :uuid do |t|
      t.references :team, null: false, type: :uuid, foreign_key: true
      t.references :data_source, null: false, type: :uuid, foreign_key: true
      t.references :league, null: false, type: :uuid, foreign_key: true

      t.string :external_code, null: false
      t.boolean :active, null: false, default: true

      t.timestamps
    end

    add_index :team_identifiers, [:data_source_id, :league_id, :external_code], unique: true, name: "index_team_identifiers_on_source_league_and_code"

    add_check_constraint :team_identifiers, "trim(external_code) <> ''", name: "check_external_code_not_blank"
  end
end
