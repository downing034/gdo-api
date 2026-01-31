# frozen_string_literal: true

module Ncaam
  module PythonPredictor
    VENV_PYTHON = Rails.root.join('db', 'data', 'ncaam', 'venv', 'bin', 'python').to_s
    MODELS_BASE_DIR = Rails.root.join('db', 'data', 'ncaam', 'models').to_s

    class PredictionError < StandardError; end

    def run_prediction(away_code:, home_code:, model_version: 'v1', neutral: false)
      script = File.join(MODELS_BASE_DIR, model_version, 'predict.py')

      unless File.exist?(script)
        raise PredictionError, "Predict script not found: #{script}"
      end

      args = ["--away", away_code, "--home", home_code]
      args += ["--neutral"] if neutral

      result = `#{VENV_PYTHON} #{script} #{args.join(' ')}`

      unless $?.success?
        raise PredictionError, "Prediction script failed for #{away_code} @ #{home_code}"
      end

      JSON.parse(result)
    rescue JSON::ParserError
      raise PredictionError, "Invalid JSON response for #{away_code} @ #{home_code}"
    end
  end
end