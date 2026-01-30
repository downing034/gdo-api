class CreateVenues < ActiveRecord::Migration[8.0]
  def change
    create_table :venues, id: :uuid do |t|
      t.string :name
      t.string :city
      t.string :region
      t.string :country
      t.integer :capacity
      t.string :surface
      t.boolean :indoor
      t.boolean :is_active
      t.decimal :latitude,  precision: 10, scale: 6
      t.decimal :longitude, precision: 10, scale: 6

      t.timestamps
    end    
  end
end
