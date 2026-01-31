# frozen_string_literal: true

module Api
  module V1
    class LeaguesController < BaseController
      def index
        data = cached("leagues/active", expires_in: 1.hour) do
          leagues = League.where(active: true).includes(:sport).order(:name)
          { leagues: leagues.map { |league| LeagueSerializer.new(league).as_json } }
        end

        render json: data
      end

      def show
        data = cached("leagues/#{params[:code].downcase}", expires_in: 1.hour) do
          league = League.includes(:sport).find_by!(code: params[:code].downcase)
          { league: LeagueSerializer.new(league).as_json }
        end

        render json: data
      end
    end
  end
end