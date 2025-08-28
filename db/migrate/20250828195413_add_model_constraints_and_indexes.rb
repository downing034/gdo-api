class AddModelConstraintsAndIndexes < ActiveRecord::Migration[8.0]
  def change
    # Sports
    change_column_null :sports, :code, false
    change_column_null :sports, :name, false
    add_index :sports, :code, unique: true

    # Leagues
    change_column_null :leagues, :code, false
    change_column_null :leagues, :name, false
    change_column_null :leagues, :active, false
    add_index :leagues, [:sport_id, :code], unique: true

    # Teams
    change_column_null :teams, :code, false
    change_column_null :teams, :location_name, false
    change_column_null :teams, :nickname, false
    change_column_null :teams, :active, false
    add_index :teams, [:league_id, :code], unique: true

    # Venues
    change_column_null :venues, :name, false
    change_column_null :venues, :is_active, false
    add_index :venues, :name, unique: true
  end
end
