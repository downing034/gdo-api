module Ncaam
  class PredictService
    VENV_PYTHON = Rails.root.join('db', 'data', 'ncaam', 'venv', 'bin', 'python').to_s
    MODELS_DIR = Rails.root.join('db', 'data', 'ncaam', 'models', 'v1').to_s
    
    attr_reader :results
    
    def initialize(model_version: 'v1', start_date: nil, end_date: nil, include_completed_games: false)
      @model_version = model_version
      @start_date = start_date
      @end_date = end_date
      @include_completed_games = include_completed_games
      @results = { created: 0, updated: 0, skipped: 0, errors: [] }
    end
    
    def call
      generate_predictions
      results
    end
    
    private
    
    def generate_predictions
      league = League.find_by!(code: 'ncaam')
      data_source = DataSource.find_by!(code: 'gdo')
      predict_script = File.join(MODELS_DIR, 'predict.py')
      
      games = fetch_games(league)
      
      games.each do |game|
        predict_game(game, predict_script, data_source)
      end
    end
    
    def fetch_games(league)
      games = Game.where(league: league).includes(:home_team, :away_team)
      
      # Apply date filters
      games = if @start_date && @end_date
        start_time_begin = @start_date.in_time_zone('America/Denver').beginning_of_day.utc
        start_time_end = @end_date.in_time_zone('America/Denver').end_of_day.utc
        games.where(start_time: start_time_begin..start_time_end)
      elsif @start_date
        games.for_date(@start_date)
      else
        games.where('start_time >= ?', Date.current.in_time_zone('America/Denver').beginning_of_day.utc)
      end
      
      # Exclude completed games unless flag is set
      unless @include_completed_games
        games = games.where.not(id: GameResult.where(final: true).select(:game_id))
      end
      
      games
    end
    
    def predict_game(game, predict_script, data_source)
      away_code = game.away_team.code
      home_code = game.home_team.code
      
      result = `#{VENV_PYTHON} #{predict_script} --away #{away_code} --home #{home_code}`
      
      unless $?.success?
        @results[:errors] << "#{away_code} @ #{home_code}: prediction failed"
        return
      end
      
      prediction_data = JSON.parse(result)
      
      away_score = prediction_data['away_team']['predicted_score']
      home_score = prediction_data['home_team']['predicted_score']
      winner_code = prediction_data['favorite']
      winner = winner_code == home_code ? game.home_team : game.away_team
      
      prediction = GamePrediction.find_or_initialize_by(
        game: game,
        model_version: @model_version,
        data_source: data_source
      )
      
      was_new = prediction.new_record?
      
      if was_new
        prediction.update!(
          away_predicted_score: away_score,
          home_predicted_score: home_score,
          predicted_winner: winner,
          generated_at: Time.current
        )
        @results[:created] += 1
      elsif prediction.away_predicted_score != away_score || 
            prediction.home_predicted_score != home_score ||
            prediction.predicted_winner_id != winner.id
        prediction.update!(
          away_predicted_score: away_score,
          home_predicted_score: home_score,
          predicted_winner: winner,
          generated_at: Time.current
        )
        @results[:updated] += 1
      else
        @results[:skipped] += 1
      end
    rescue JSON::ParserError
      @results[:errors] << "#{away_code} @ #{home_code}: invalid JSON"
    rescue => e
      @results[:errors] << "#{away_code} @ #{home_code}: #{e.message}"
    end
  end
end