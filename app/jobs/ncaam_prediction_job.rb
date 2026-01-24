class NcaamPredictionJob < ApplicationJob
  queue_as :default

  def perform
    results = Ncaam::PredictionService.new.call
    
    Rails.logger.info "NCAAM Predictions: created=#{results[:created]} updated=#{results[:updated]} skipped=#{results[:skipped]} errors=#{results[:errors].count}"
  end
end