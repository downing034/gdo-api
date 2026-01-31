# frozen_string_literal: true

module Ncaam
  class SimulateService
    include PythonPredictor

    def initialize(away_team_code:, home_team_code:, neutral: false, model_version: 'v1')
      @away_team_code = away_team_code.upcase
      @home_team_code = home_team_code.upcase
      @neutral = neutral
      @model_version = model_version
    end

    def call
      validate_teams!
      result = run_prediction(
        away_code: @away_team_code,
        home_code: @home_team_code,
        model_version: @model_version,
        neutral: @neutral
      )
      build_response(result)
    end

    private

    def validate_teams!
      league = League.find_by!(code: 'ncaam')
      @away_team = league.teams.find_by!(code: @away_team_code)
      @home_team = league.teams.find_by!(code: @home_team_code)
    end

    def build_response(result)
      {
        away_team: {
          code: @away_team.code,
          location_name: @away_team.location_name,
          nickname: @away_team.nickname,
          predicted_score: result['away_team']['predicted_score']
        },
        home_team: {
          code: @home_team.code,
          location_name: @home_team.location_name,
          nickname: @home_team.nickname,
          predicted_score: result['home_team']['predicted_score']
        },
        spread: result['spread'],
        favorite: result['favorite'],
        total: result['total']
      }
    end
  end
end