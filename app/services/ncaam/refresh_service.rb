module Ncaam
  class RefreshService
    VENV_PYTHON = Rails.root.join('db', 'data', 'ncaam', 'venv', 'bin', 'python').to_s
    MODELS_DIR = Rails.root.join('db', 'data', 'ncaam', 'models', 'v1').to_s
    
    attr_reader :results
    
    def initialize
      @results = { processed: false, trained: false }
    end
    
    def call
      process_data
      train_model
      results
    end
    
    def process_data
      DataProcessorService.new.call
      @results[:processed] = true
    end
    
    def train_model
      train_script = File.join(MODELS_DIR, 'train_model.py')
      success = system("#{VENV_PYTHON} #{train_script}")
      raise "Training failed" unless success
      @results[:trained] = true
    end
  end
end