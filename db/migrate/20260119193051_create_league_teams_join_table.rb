class CreateLeagueTeamsJoinTable < ActiveRecord::Migration[8.0]
  def change
    create_join_table :leagues, :teams, column_options: { type: :uuid } do |t|
      t.index [:league_id, :team_id], unique: true
      t.index [:team_id, :league_id]
    end
  end
end