# frozen_string_literal: true

module Api
  module V1
    class PredictionsController < BaseController
      LEAGUE_SERVICES = {
        'ncaam' => Ncaam::SimulateService
      }.freeze

      rescue_from Ncaam::PythonPredictor::PredictionError, with: :prediction_failed

      def simulate
        cache_key = "simulate/#{Digest::MD5.hexdigest(simulate_params.to_json)}"

        data = cached(cache_key, expires_in: 15.minutes) do
          service = build_service
          result = service.call
          { prediction: result }
        end

        render json: data
      end

      private

      def prediction_failed(exception)
        render_error(
          code: "prediction_failed",
          message: exception.message,
          status: :unprocessable_entity
        )
      end

      def build_service
        league_code = simulate_params[:league]&.downcase

        raise ArgumentError, "League is required" if league_code.blank?

        service_class = LEAGUE_SERVICES[league_code]
        raise ArgumentError, "League '#{league_code}' not supported" unless service_class

        service_class.new(
          away_team_code: simulate_params[:away_team_code],
          home_team_code: simulate_params[:home_team_code],
          neutral: simulate_params[:neutral] || false
        )
      end

      def simulate_params
        params.permit(:league, :away_team_code, :home_team_code, :neutral)
      end
    end
  end
end