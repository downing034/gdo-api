# frozen_string_literal: true

module Api
  module V1
    class GamesController < BaseController
      def index
        cache_key = "games/#{params[:league]}/#{params[:date]}/#{params[:status]}"

        data = cached(cache_key, expires_in: 5.minutes) do
          games = games_scope
                    .includes(:league, :home_team, :away_team, :season)
                    .order(:start_time)
          {
            games: games.map { |game| GameSerializer.new(game).as_json },
            meta: {
              count: games.size,
              filters: applied_filters
            }
          }
        end

        render json: data
      end

      def show
        data = cached("games/#{params[:id]}", expires_in: 1.minute) do
          game = Game.includes(
            :league, :home_team, :away_team, :season,
            :game_result, :game_odds, game_predictions: :data_source
          ).find(params[:id])

          {
            game: GameSerializer.new(game, include_result: true, include_odds: true, include_predictions: true).as_json
          }
        end

        render json: data
      end

      private

      def games_scope
        scope = Game.all
        scope = scope.where(league: League.find_by!(code: params[:league])) if params[:league].present?
        scope = scope.where(start_time: date_range) if params[:date].present?
        scope = scope.where(status: params[:status]) if params[:status].present?
        scope
      end

      def date_range
        date = Date.parse(params[:date])
        date.beginning_of_day..date.end_of_day
      end

      def applied_filters
        {
          league: params[:league],
          date: params[:date],
          status: params[:status]
        }.compact
      end
    end
  end
end