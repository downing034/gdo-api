# frozen_string_literal: true

module Api
  module V1
    class TeamsController < BaseController
      def show
        data = cached("teams/#{params[:code].upcase}", expires_in: 15.minutes) do
          team = Team.find_by!(code: params[:code].upcase)
          { team: TeamSerializer.new(team).as_json }
        end

        render json: data
      end
    end
  end
end