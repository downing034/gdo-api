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

ActiveRecord::Schema[8.0].define(version: 2026_01_27_180940) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"
  enable_extension "pgcrypto"

  create_table "basketball_game_player_stats", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "game_id", null: false
    t.uuid "player_id", null: false
    t.uuid "team_id", null: false
    t.integer "minutes_played"
    t.integer "points"
    t.integer "field_goals_made"
    t.integer "field_goals_attempted"
    t.integer "three_pointers_made"
    t.integer "three_pointers_attempted"
    t.integer "free_throws_made"
    t.integer "free_throws_attempted"
    t.integer "offensive_rebounds"
    t.integer "defensive_rebounds"
    t.integer "assists"
    t.integer "steals"
    t.integer "blocks"
    t.integer "turnovers"
    t.integer "fouls"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game_id", "player_id"], name: "index_basketball_game_player_stats_on_game_id_and_player_id", unique: true
    t.index ["game_id"], name: "index_basketball_game_player_stats_on_game_id"
    t.index ["player_id", "game_id"], name: "idx_bb_player_stats_player_game"
    t.index ["player_id"], name: "index_basketball_game_player_stats_on_player_id"
    t.index ["team_id"], name: "index_basketball_game_player_stats_on_team_id"
  end

  create_table "basketball_game_team_stats", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "game_id", null: false
    t.uuid "team_id", null: false
    t.integer "field_goals_made"
    t.integer "field_goals_attempted"
    t.integer "three_pointers_made"
    t.integer "three_pointers_attempted"
    t.integer "free_throws_made"
    t.integer "free_throws_attempted"
    t.integer "offensive_rebounds"
    t.integer "defensive_rebounds"
    t.integer "assists"
    t.integer "steals"
    t.integer "blocks"
    t.integer "turnovers"
    t.integer "fouls"
    t.integer "points_off_turnovers"
    t.integer "fast_break_points"
    t.integer "points_in_paint"
    t.integer "largest_lead"
    t.integer "lead_changes"
    t.decimal "time_leading_pct", precision: 5, scale: 2
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game_id", "team_id"], name: "index_basketball_game_team_stats_on_game_id_and_team_id", unique: true
    t.index ["game_id"], name: "index_basketball_game_team_stats_on_game_id"
    t.index ["team_id"], name: "index_basketball_game_team_stats_on_team_id"
  end

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
    t.boolean "is_opening", default: false, null: false
    t.index ["data_source_id"], name: "index_game_odds_on_data_source_id"
    t.index ["game_id", "fetched_at"], name: "index_game_odds_on_game_id_and_fetched_at"
    t.index ["game_id"], name: "index_game_odds_on_game_id"
    t.index ["moneyline_favorite_team_id"], name: "index_game_odds_on_moneyline_favorite_team_id"
    t.index ["spread_favorite_team_id"], name: "index_game_odds_on_spread_favorite_team_id"
  end

  create_table "game_predictions", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "game_id", null: false
    t.string "model_version", null: false
    t.uuid "data_source_id", null: false
    t.decimal "away_predicted_score"
    t.decimal "home_predicted_score"
    t.uuid "predicted_winner_id"
    t.decimal "confidence"
    t.datetime "generated_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["data_source_id"], name: "index_game_predictions_on_data_source_id"
    t.index ["game_id", "model_version", "data_source_id", "generated_at"], name: "index_game_predictions_unique", unique: true
    t.index ["game_id", "model_version"], name: "index_game_predictions_on_game_id_and_model_version"
    t.index ["game_id"], name: "index_game_predictions_on_game_id"
    t.index ["predicted_winner_id"], name: "index_game_predictions_on_predicted_winner_id"
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
    t.integer "period"
    t.index ["game_id"], name: "index_game_results_on_game_id", unique: true
  end

  create_table "games", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.uuid "league_id", null: false
    t.uuid "home_team_id", null: false
    t.uuid "away_team_id", null: false
    t.datetime "start_time"
    t.string "status", default: "scheduled"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.uuid "season_id", null: false
    t.string "external_id"
    t.boolean "is_stale", default: false
    t.index ["away_team_id"], name: "index_games_on_away_team_id"
    t.index ["home_team_id"], name: "index_games_on_home_team_id"
    t.index ["league_id", "external_id"], name: "index_games_on_league_id_and_external_id", unique: true
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

  create_table "players", id: :uuid, default: -> { "gen_random_uuid()" }, force: :cascade do |t|
    t.string "external_id", null: false
    t.uuid "data_source_id", null: false
    t.uuid "team_id"
    t.string "name", null: false
    t.string "position"
    t.boolean "active", default: true, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["data_source_id", "external_id"], name: "index_players_on_data_source_id_and_external_id", unique: true
    t.index ["data_source_id"], name: "index_players_on_data_source_id"
    t.index ["team_id"], name: "index_players_on_team_id"
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

  create_table "solid_queue_blocked_executions", force: :cascade do |t|
    t.bigint "job_id", null: false
    t.string "queue_name", null: false
    t.integer "priority", default: 0, null: false
    t.string "concurrency_key", null: false
    t.datetime "expires_at", null: false
    t.datetime "created_at", null: false
    t.index ["concurrency_key", "priority", "job_id"], name: "index_solid_queue_blocked_executions_for_release"
    t.index ["expires_at", "concurrency_key"], name: "index_solid_queue_blocked_executions_for_maintenance"
    t.index ["job_id"], name: "index_solid_queue_blocked_executions_on_job_id", unique: true
  end

  create_table "solid_queue_claimed_executions", force: :cascade do |t|
    t.bigint "job_id", null: false
    t.bigint "process_id"
    t.datetime "created_at", null: false
    t.index ["job_id"], name: "index_solid_queue_claimed_executions_on_job_id", unique: true
    t.index ["process_id", "job_id"], name: "index_solid_queue_claimed_executions_on_process_id_and_job_id"
  end

  create_table "solid_queue_failed_executions", force: :cascade do |t|
    t.bigint "job_id", null: false
    t.text "error"
    t.datetime "created_at", null: false
    t.index ["job_id"], name: "index_solid_queue_failed_executions_on_job_id", unique: true
  end

  create_table "solid_queue_jobs", force: :cascade do |t|
    t.string "queue_name", null: false
    t.string "class_name", null: false
    t.text "arguments"
    t.integer "priority", default: 0, null: false
    t.string "active_job_id"
    t.datetime "scheduled_at"
    t.datetime "finished_at"
    t.string "concurrency_key"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["active_job_id"], name: "index_solid_queue_jobs_on_active_job_id"
    t.index ["class_name"], name: "index_solid_queue_jobs_on_class_name"
    t.index ["finished_at"], name: "index_solid_queue_jobs_on_finished_at"
    t.index ["queue_name", "finished_at"], name: "index_solid_queue_jobs_for_filtering"
    t.index ["scheduled_at", "finished_at"], name: "index_solid_queue_jobs_for_alerting"
  end

  create_table "solid_queue_pauses", force: :cascade do |t|
    t.string "queue_name", null: false
    t.datetime "created_at", null: false
    t.index ["queue_name"], name: "index_solid_queue_pauses_on_queue_name", unique: true
  end

  create_table "solid_queue_processes", force: :cascade do |t|
    t.string "kind", null: false
    t.datetime "last_heartbeat_at", null: false
    t.bigint "supervisor_id"
    t.integer "pid", null: false
    t.string "hostname"
    t.text "metadata"
    t.datetime "created_at", null: false
    t.string "name", null: false
    t.index ["last_heartbeat_at"], name: "index_solid_queue_processes_on_last_heartbeat_at"
    t.index ["name", "supervisor_id"], name: "index_solid_queue_processes_on_name_and_supervisor_id", unique: true
    t.index ["supervisor_id"], name: "index_solid_queue_processes_on_supervisor_id"
  end

  create_table "solid_queue_ready_executions", force: :cascade do |t|
    t.bigint "job_id", null: false
    t.string "queue_name", null: false
    t.integer "priority", default: 0, null: false
    t.datetime "created_at", null: false
    t.index ["job_id"], name: "index_solid_queue_ready_executions_on_job_id", unique: true
    t.index ["priority", "job_id"], name: "index_solid_queue_poll_all"
    t.index ["queue_name", "priority", "job_id"], name: "index_solid_queue_poll_by_queue"
  end

  create_table "solid_queue_recurring_executions", force: :cascade do |t|
    t.bigint "job_id", null: false
    t.string "task_key", null: false
    t.datetime "run_at", null: false
    t.datetime "created_at", null: false
    t.index ["job_id"], name: "index_solid_queue_recurring_executions_on_job_id", unique: true
    t.index ["task_key", "run_at"], name: "index_solid_queue_recurring_executions_on_task_key_and_run_at", unique: true
  end

  create_table "solid_queue_recurring_tasks", force: :cascade do |t|
    t.string "key", null: false
    t.string "schedule", null: false
    t.string "command", limit: 2048
    t.string "class_name"
    t.text "arguments"
    t.string "queue_name"
    t.integer "priority", default: 0
    t.boolean "static", default: true, null: false
    t.text "description"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["key"], name: "index_solid_queue_recurring_tasks_on_key", unique: true
    t.index ["static"], name: "index_solid_queue_recurring_tasks_on_static"
  end

  create_table "solid_queue_scheduled_executions", force: :cascade do |t|
    t.bigint "job_id", null: false
    t.string "queue_name", null: false
    t.integer "priority", default: 0, null: false
    t.datetime "scheduled_at", null: false
    t.datetime "created_at", null: false
    t.index ["job_id"], name: "index_solid_queue_scheduled_executions_on_job_id", unique: true
    t.index ["scheduled_at", "priority", "job_id"], name: "index_solid_queue_dispatch_all"
  end

  create_table "solid_queue_semaphores", force: :cascade do |t|
    t.string "key", null: false
    t.integer "value", default: 1, null: false
    t.datetime "expires_at", null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["expires_at"], name: "index_solid_queue_semaphores_on_expires_at"
    t.index ["key", "value"], name: "index_solid_queue_semaphores_on_key_and_value"
    t.index ["key"], name: "index_solid_queue_semaphores_on_key", unique: true
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

  add_foreign_key "basketball_game_player_stats", "games"
  add_foreign_key "basketball_game_player_stats", "players"
  add_foreign_key "basketball_game_player_stats", "teams"
  add_foreign_key "basketball_game_team_stats", "games"
  add_foreign_key "basketball_game_team_stats", "teams"
  add_foreign_key "game_odds", "data_sources"
  add_foreign_key "game_odds", "games"
  add_foreign_key "game_odds", "teams", column: "moneyline_favorite_team_id"
  add_foreign_key "game_odds", "teams", column: "spread_favorite_team_id"
  add_foreign_key "game_predictions", "data_sources"
  add_foreign_key "game_predictions", "games"
  add_foreign_key "game_predictions", "teams", column: "predicted_winner_id"
  add_foreign_key "game_results", "games"
  add_foreign_key "games", "leagues"
  add_foreign_key "games", "seasons"
  add_foreign_key "games", "teams", column: "away_team_id"
  add_foreign_key "games", "teams", column: "home_team_id"
  add_foreign_key "leagues", "sports"
  add_foreign_key "players", "data_sources"
  add_foreign_key "players", "teams"
  add_foreign_key "seasons", "leagues"
  add_foreign_key "solid_queue_blocked_executions", "solid_queue_jobs", column: "job_id", on_delete: :cascade
  add_foreign_key "solid_queue_claimed_executions", "solid_queue_jobs", column: "job_id", on_delete: :cascade
  add_foreign_key "solid_queue_failed_executions", "solid_queue_jobs", column: "job_id", on_delete: :cascade
  add_foreign_key "solid_queue_ready_executions", "solid_queue_jobs", column: "job_id", on_delete: :cascade
  add_foreign_key "solid_queue_recurring_executions", "solid_queue_jobs", column: "job_id", on_delete: :cascade
  add_foreign_key "solid_queue_scheduled_executions", "solid_queue_jobs", column: "job_id", on_delete: :cascade
  add_foreign_key "team_identifiers", "data_sources"
  add_foreign_key "team_identifiers", "leagues"
  add_foreign_key "team_identifiers", "teams"
end
