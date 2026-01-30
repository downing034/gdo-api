# app/jobs/ncaam_prediction_job.rb
class NcaamPredictionJob < ApplicationJob
  queue_as :default

  # Run predictions for all model versions
  MODEL_VERSIONS = %w[v1 v2].freeze

  def perform(model_versions: MODEL_VERSIONS)
    versions = Array(model_versions)
    
    versions.each do |version|
      run_predictions(version)
    end
  end

  private

  def run_predictions(model_version)
    results = Ncaam::PredictService.new(model_version: model_version).call
    
    Rails.logger.info(
      "NCAAM Predictions (#{model_version}): " \
      "created=#{results[:created]} " \
      "updated=#{results[:updated]} " \
      "skipped=#{results[:skipped]} " \
      "errors=#{results[:errors].count}"
    )
    
    if results[:errors].any?
      Rails.logger.warn("NCAAM Prediction Errors (#{model_version}): #{results[:errors].first(5).join(', ')}")
    end
  end
end