# frozen_string_literal: true

module Api
  module V1
    module Leagues
      class TeamsController < BaseController
        def index
          league_code = params[:league_code].downcase

          data = cached("leagues/#{league_code}/teams", expires_in: 1.hour) do
            league = League.find_by!(code: league_code)
            teams = league.teams.where(active: true).order(:location_name)
            {
              league: league.code,
              teams: teams.map { |team| TeamSerializer.new(team).as_json }
            }
          end

          render json: data
        end
      end
    end
  end
end