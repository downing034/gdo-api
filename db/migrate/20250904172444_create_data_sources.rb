class CreateDataSources < ActiveRecord::Migration[8.0]
  def change
    create_table :data_sources, id: :uuid do |t|
      t.string :code
      t.string :name
      t.string :base_url

      t.timestamps
    end
  end
end
