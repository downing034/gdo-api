class ExportStaleGamesJob < ApplicationJob
  queue_as :default

  def perform(league_code)
    league = League.find_by!(code: league_code)
    
    stale_games = Game.includes(:home_team, :away_team)
                      .where(league: league, is_stale: true)
                      .order(:start_time)
    
    if stale_games.empty?
      puts "No stale games found for #{league_code}"
      return
    end
    
    dir = "db/data/#{league_code}/stale"
    FileUtils.mkdir_p(dir)
    
    filename = "#{dir}/stale_games_#{Date.current.strftime('%Y%m%d')}.csv"
    
    CSV.open(filename, 'w') do |csv|
      csv << ['id', 'external_id', 'game_date', 'start_time', 'away_team', 'home_team', 'status', 'created_at']
      
      stale_games.each do |game|
        csv << [
          game.id,
          game.external_id,
          game.game_date,
          game.start_time&.in_time_zone('America/Denver')&.strftime('%H:%M'),
          game.away_team.code,
          game.home_team.code,
          game.status,
          game.created_at.strftime('%Y-%m-%d %H:%M')
        ]
      end
    end
    
    Rails.logger.info "Exported #{stale_games.count} stale #{league_code} games to #{filename}"
  end
end