# frozen_string_literal: true

module Api
  module V1
    class BaseController < ApplicationController
      rescue_from StandardError, with: :internal_error
      rescue_from ActiveRecord::RecordNotFound, with: :not_found
      rescue_from ActiveRecord::RecordInvalid, with: :unprocessable_entity
      rescue_from ActionController::ParameterMissing, with: :bad_request
      rescue_from ArgumentError, with: :bad_request

      private

      def not_found(exception)
        render_error(
          code: "not_found",
          message: exception.message,
          status: :not_found
        )
      end

      def bad_request(exception)
        render_error(
          code: "bad_request",
          message: exception.message,
          status: :bad_request
        )
      end

      def unprocessable_entity(exception)
        render_error(
          code: "unprocessable_entity",
          message: exception.message,
          details: exception.record&.errors&.to_hash,
          status: :unprocessable_entity
        )
      end

      def internal_error(exception)
        Rails.logger.error("Internal error: #{exception.message}")
        Rails.logger.error(exception.backtrace.join("\n"))

        render_error(
          code: "internal_error",
          message: Rails.env.production? ? "Something went wrong" : exception.message,
          status: :internal_server_error
        )
      end

      def render_error(code:, message:, status:, details: nil)
        body = {
          error: {
            code: code,
            message: message
          }
        }
        body[:error][:details] = details if details.present?

        render json: body, status: status
      end

      def cache_key(*parts)
        parts.join("/")
      end

      def cached(key, expires_in:)
        Rails.cache.fetch(key, expires_in: expires_in) { yield }
      end
    end
  end
end