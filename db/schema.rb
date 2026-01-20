# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.0].define(version: 2026_01_19_193051) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"
  enable_extension "pgcrypto"

  create_table "data_sources", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.string "code"
    t.string "name"
    t.string "base_url"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "game_odds", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "game_id", null: false
    t.uuid "spread_favorite_team_id"
    t.decimal "spread_value"
    t.integer "spread_favorite_odds"
    t.integer "spread_underdog_odds"
    t.decimal "total_line"
    t.integer "over_odds"
    t.integer "under_odds"
    t.uuid "moneyline_favorite_team_id"
    t.integer "moneyline_favorite_odds"
    t.integer "moneyline_underdog_odds"
    t.uuid "data_source_id", null: false
    t.datetime "fetched_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["data_source_id"], name: "index_game_odds_on_data_source_id"
    t.index ["game_id", "fetched_at"], name: "index_game_odds_on_game_id_and_fetched_at"
    t.index ["game_id"], name: "index_game_odds_on_game_id"
    t.index ["moneyline_favorite_team_id"], name: "index_game_odds_on_moneyline_favorite_team_id"
    t.index ["spread_favorite_team_id"], name: "index_game_odds_on_spread_favorite_team_id"
  end

  create_table "game_results", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "game_id", null: false
    t.integer "home_score"
    t.integer "away_score"
    t.boolean "final", default: false
    t.json "period_scores"
    t.text "notes"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game_id"], name: "index_game_results_on_game_id", unique: true
  end

  create_table "games", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "league_id", null: false
    t.date "game_date", null: false
    t.uuid "home_team_id", null: false
    t.uuid "away_team_id", null: false
    t.datetime "start_time"
    t.integer "status", default: 0
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.uuid "season_id", null: false
    t.index ["away_team_id"], name: "index_games_on_away_team_id"
    t.index ["home_team_id"], name: "index_games_on_home_team_id"
    t.index ["league_id", "game_date", "home_team_id", "away_team_id", "start_time"], name: "index_games_on_unique_game", unique: true
    t.index ["league_id", "game_date"], name: "index_games_on_league_id_and_game_date"
    t.index ["league_id"], name: "index_games_on_league_id"
    t.index ["season_id"], name: "index_games_on_season_id"
    t.check_constraint "home_team_id <> away_team_id", name: "games_different_teams"
  end

  create_table "leagues", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "sport_id", null: false
    t.string "code", null: false
    t.string "name", null: false
    t.string "display_name"
    t.boolean "has_conferences"
    t.boolean "active", null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["sport_id", "code"], name: "index_leagues_on_sport_id_and_code", unique: true
    t.index ["sport_id"], name: "index_leagues_on_sport_id"
  end

  create_table "leagues_teams", id: false, force: :cascade do |t|
    t.uuid "league_id", null: false
    t.uuid "team_id", null: false
    t.index ["league_id", "team_id"], name: "index_leagues_teams_on_league_id_and_team_id", unique: true
    t.index ["team_id", "league_id"], name: "index_leagues_teams_on_team_id_and_league_id"
  end

  create_table "seasons", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "league_id", null: false
    t.string "name", null: false
    t.date "start_date", null: false
    t.date "end_date", null: false
    t.boolean "active", default: false, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["active"], name: "index_seasons_on_active"
    t.index ["league_id", "name"], name: "index_seasons_on_league_id_and_name", unique: true
    t.index ["league_id"], name: "index_seasons_on_league_id"
  end

  create_table "sports", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.string "code", null: false
    t.string "name", null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["code"], name: "index_sports_on_code", unique: true
  end

  create_table "team_identifiers", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "team_id", null: false
    t.uuid "data_source_id", null: false
    t.uuid "league_id", null: false
    t.string "external_code", null: false
    t.boolean "active", default: true, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["data_source_id", "league_id", "external_code"], name: "index_team_identifiers_on_source_league_and_code", unique: true
    t.index ["data_source_id"], name: "index_team_identifiers_on_data_source_id"
    t.index ["league_id"], name: "index_team_identifiers_on_league_id"
    t.index ["team_id"], name: "index_team_identifiers_on_team_id"
    t.check_constraint "TRIM(BOTH FROM external_code) <> ''::text", name: "check_external_code_not_blank"
  end

  create_table "teams", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.string "code", null: false
    t.string "location_name", null: false
    t.string "nickname", null: false
    t.boolean "active", null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["code"], name: "index_teams_on_code", unique: true
  end

  create_table "venues", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.string "name", null: false
    t.string "city"
    t.string "region"
    t.string "country"
    t.integer "capacity"
    t.string "surface"
    t.boolean "indoor"
    t.boolean "is_active", null: false
    t.decimal "latitude", precision: 10, scale: 6
    t.decimal "longitude", precision: 10, scale: 6
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["name"], name: "index_venues_on_name", unique: true
  end

  add_foreign_key "game_odds", "data_sources"
  add_foreign_key "game_odds", "games"
  add_foreign_key "game_odds", "teams", column: "moneyline_favorite_team_id"
  add_foreign_key "game_odds", "teams", column: "spread_favorite_team_id"
  add_foreign_key "game_results", "games"
  add_foreign_key "games", "leagues"
  add_foreign_key "games", "seasons"
  add_foreign_key "games", "teams", column: "away_team_id"
  add_foreign_key "games", "teams", column: "home_team_id"
  add_foreign_key "leagues", "sports"
  add_foreign_key "seasons", "leagues"
  add_foreign_key "team_identifiers", "data_sources"
  add_foreign_key "team_identifiers", "leagues"
  add_foreign_key "team_identifiers", "teams"
end
