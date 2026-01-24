class EspnFetchGamesJob < ApplicationJob
  queue_as :default

  LEAGUE_TIERS = {
    'ncaam' => {
      'hourly' => [0],
      'every_4h' => [1],
      'twice_daily' => [2],
      'daily' => [3, 4, 5],
      'yesterday_cleanup' => [-1]
    },
    'nfl' => {
      'hourly' => [0],
      'every_4h' => [1],
      'twice_daily' => [2],
      'daily' => [3, 4, 5, 6, 7],
      'yesterday_cleanup' => [-1]
    },
    # 'mlb' => {
    #   'hourly' => [0],
    #   'every_4h' => [1],
    #   'twice_daily' => [],
    #   'daily' => [2, 3],
    #   'yesterday_cleanup' => [-1]
    # }
  }.freeze

  def perform(tier)
    LEAGUE_TIERS.each do |league_code, tiers|
      offsets = tiers[tier] || []
      offsets.each do |offset|
        date = Date.current + offset.days
        fetch_for_date(league_code, date)
      end
    end
  end

  def self.fetch(league_code, date = nil)
    date = date ? Date.parse(date.to_s) : Date.current
    new.fetch_for_date(league_code, date)
  end

  def fetch_for_date(league_code, date)
    service = Espn::GamesService.new(league_code: league_code, date: date)
    results = service.call

    Rails.logger.info "[EspnFetchGamesJob] #{league_code.upcase} #{date}: #{results[:games_created]} created, #{results[:games_updated]} updated, #{results[:odds_created]} odds created"
    results
  end
end