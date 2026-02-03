module Ncaam
  class PredictService
    VENV_PYTHON = Rails.root.join('db', 'data', 'ncaam', 'venv', 'bin', 'python').to_s
    MODELS_BASE_DIR = Rails.root.join('db', 'data', 'ncaam', 'models').to_s
    
    # Model version configurations
    MODEL_CONFIGS = {
      'v1' => {
        script: 'predict.py',
        data_source_code: 'gdo'
      },
      'v2' => {
        script: 'predict.py',
        data_source_code: 'gdo'
      },
      'v3' => {
        script: 'predict.py',
        data_source_code: 'gdo'
      }
    }.freeze
    
    attr_reader :results
    
    def initialize(model_version: 'v1', start_date: nil, end_date: nil, include_completed_games: false)
      @model_version = model_version
      @start_date = start_date
      @end_date = end_date
      @include_completed_games = include_completed_games
      @results = { created: 0, updated: 0, skipped: 0, errors: [] }
      
      validate_model_version!
    end
    
    def call
      generate_predictions
      results
    end
    
    private
    
    def validate_model_version!
      unless MODEL_CONFIGS.key?(@model_version)
        raise ArgumentError, "Unknown model version: #{@model_version}. Valid versions: #{MODEL_CONFIGS.keys.join(', ')}"
      end
    end
    
    def model_config
      MODEL_CONFIGS[@model_version]
    end
    
    def models_dir
      File.join(MODELS_BASE_DIR, @model_version)
    end
    
    def predict_script
      File.join(models_dir, model_config[:script])
    end
    
    def generate_predictions
      league = League.find_by!(code: 'ncaam')
      data_source = DataSource.find_by!(code: model_config[:data_source_code])
      
      unless File.exist?(predict_script)
        @results[:errors] << "Predict script not found: #{predict_script}"
        return
      end
      
      games = fetch_games(league)
      
      games.each do |game|
        predict_game(game, data_source)
      end
    end
    
    def fetch_games(league)
      games = Game.where(league: league).includes(:home_team, :away_team)
      
      # Apply date filters using start_time column
      games = if @start_date && @end_date
        start_time_begin = @start_date.in_time_zone('America/Denver').beginning_of_day.utc
        start_time_end = @end_date.in_time_zone('America/Denver').end_of_day.utc
        games.where(start_time: start_time_begin..start_time_end)
      elsif @start_date
        start_time_begin = @start_date.in_time_zone('America/Denver').beginning_of_day.utc
        start_time_end = @start_date.in_time_zone('America/Denver').end_of_day.utc
        games.where(start_time: start_time_begin..start_time_end)
      else
        games.where('start_time >= ?', Date.current.in_time_zone('America/Denver').beginning_of_day.utc)
      end
      
      # Exclude completed games unless flag is set
      unless @include_completed_games
        games = games.where.not(id: GameResult.where(final: true).select(:game_id))
      end
      
      games
    end
    
    def predict_game(game, data_source)
      away_code = game.away_team.code
      home_code = game.home_team.code
      
      # Build command - same interface for v1 and v2
      cmd = "#{VENV_PYTHON} #{predict_script} --away #{away_code} --home #{home_code}"
      
      result = `#{cmd}`
      
      unless $?.success?
        @results[:errors] << "#{away_code} @ #{home_code}: prediction failed (#{@model_version})"
        return
      end
      
      prediction_data = JSON.parse(result)
      
      # Handle error response from script
      if prediction_data['error']
        @results[:errors] << "#{away_code} @ #{home_code}: #{prediction_data['error']}"
        return
      end
      
      away_score = prediction_data['away_team']['predicted_score']
      home_score = prediction_data['home_team']['predicted_score']
      winner_code = prediction_data['favorite']
      winner = winner_code == home_code ? game.home_team : game.away_team
      
      # Determine actual model version (v2 can be v2_vegas or v2_no_vegas)
      actual_model_version = if @model_version == 'v2' && prediction_data.dig('model_info', 'model_type')
        "v2_#{prediction_data['model_info']['model_type']}"
      else
        @model_version
      end
      
      prediction = GamePrediction.find_or_initialize_by(
        game: game,
        model_version: actual_model_version,
        data_source: data_source
      )
      
      was_new = prediction.new_record?
      
      # Build attributes
      attrs = {
        away_predicted_score: away_score,
        home_predicted_score: home_score,
        predicted_winner: winner,
        generated_at: Time.current
      }
      
      if was_new
        prediction.update!(attrs)
        @results[:created] += 1
      elsif prediction.away_predicted_score != away_score || 
            prediction.home_predicted_score != home_score ||
            prediction.predicted_winner_id != winner.id
        prediction.update!(attrs)
        @results[:updated] += 1
      else
        @results[:skipped] += 1
      end
    rescue JSON::ParserError
      @results[:errors] << "#{away_code} @ #{home_code}: invalid JSON (#{@model_version})"
    rescue => e
      @results[:errors] << "#{away_code} @ #{home_code}: #{e.message}"
    end
  end
end