require 'net/http'
require 'json'

module Espn
  class GamesService
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

    def initialize(league_code:, date: Date.current)
      @date = date
      @league_code = league_code
      @league = League.find_by!(code: league_code)
      @espn_source = DataSource.find_by!(code: 'espn')
      @config = ESPN_LEAGUE_CONFIG[league_code]
      
      raise ArgumentError, "Unknown league: #{league_code}" unless @config
    end

    def call
      response = fetch_scoreboard
      return { success: false, error: "Failed to fetch scoreboard" } unless response

      events = response["events"] || []

      results = {
        games_created: 0,
        games_updated: 0,
        games_skipped: 0,
        odds_created: 0,
        odds_skipped: 0,
        results_created: 0,
        teams_created: 0,
        errors: []
      }

      events.each do |event|
        process_event(event, results)
      rescue => e
        results[:errors] << { event_id: event["id"], error: e.message }
        puts "  ❌ Error processing event #{event['id']}: #{e.message}"
      end

      espn_event_ids = events.map { |e| e["id"] }
      results[:games_marked_stale] = mark_stale_games(espn_event_ids)

      results
    end

    private

    def fetch_scoreboard
      url = build_url
      puts "Fetching: #{url}"
      
      uri = URI(url)
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = true
      http.verify_mode = OpenSSL::SSL::VERIFY_NONE
      
      request = Net::HTTP::Get.new(uri)
      response = http.request(request)
      
      puts "Response length: #{response.body.length}"
      parsed = JSON.parse(response.body)
      puts "Events count: #{parsed['events']&.length || 0}"
      parsed
    rescue => e
      puts "ESPN fetch failed: #{e.message}"
      puts e.backtrace.first(5).join("\n")
      nil
    end

    def build_url
      url = "#{BASE_URL}/#{@config[:sport]}/#{@config[:league_path]}/scoreboard"
      params = ["dates=#{@date.strftime('%Y%m%d')}"]
      params << "groups=#{@config[:groups]}" if @config[:groups]
      params << "limit=#{@config[:limit]}" if @config[:limit]
      "#{url}?#{params.join('&')}"
    end

    def process_event(event, results)
      competition = event.dig("competitions", 0)
      return unless competition

      competitors = competition["competitors"]
      home_competitor = competitors.find { |c| c["homeAway"] == "home" }
      away_competitor = competitors.find { |c| c["homeAway"] == "away" }

      return unless home_competitor && away_competitor

      home_espn_abbr = home_competitor.dig("team", "abbreviation")
      away_espn_abbr = away_competitor.dig("team", "abbreviation")

      home_team = find_or_create_team(home_espn_abbr, home_competitor["team"], results)
      away_team = find_or_create_team(away_espn_abbr, away_competitor["team"], results)

      unless home_team && away_team
        results[:errors] << {
          event_id: event["id"],
          error: "Could not resolve teams: #{away_espn_abbr} @ #{home_espn_abbr}"
        }
        return
      end

      game_datetime = Time.parse(event["date"]).utc
      game_date = game_datetime.in_time_zone('America/Denver').to_date

      season = Season.find_by(
        league: @league,
        start_date: ..game_date,
        end_date: game_date..
      )

      unless season
        results[:errors] << {
          event_id: event["id"],
          error: "No season found for date #{game_date}"
        }
        return
      end

      status_type = competition.dig("status", "type")
      game_status = map_status(status_type)
      game_state = status_type["state"]

      espn_id = event["id"]

      # First try to find by external_id
      game = Game.find_by(league: @league, external_id: espn_id)

      # Fall back to matching by teams/date if no external_id match
      if game.nil?
        existing_games = Game.where(
          league: @league,
          game_date: game_date,
          home_team: home_team,
          away_team: away_team
        )

        game = if existing_games.count == 0
          Game.new(
            league: @league,
            game_date: game_date,
            home_team: home_team,
            away_team: away_team
          )
        elsif existing_games.count == 1
          existing_games.first
        else
          # Doubleheader - find closest start time
          existing_games.min_by { |g| (g.start_time - game_datetime).abs }
        end
      end

      is_new_game = game.new_record?

      game.assign_attributes(
        season: season,
        status: game_status,
        start_time: game_datetime,
        external_id: espn_id,
        is_stale: false
      )

      game.save!

      if is_new_game
        results[:games_created] += 1
        puts "Created: #{away_team.code} @ #{home_team.code} on #{game_date} at #{game_datetime.strftime('%H:%M')} UTC"
      else
        results[:games_updated] += 1
        puts "Updated: #{away_team.code} @ #{home_team.code} on #{game_date}"
      end

      # Process odds only for pre-game status
      if game_state == "pre"
        odds_data = competition.dig("odds", 0)
        if odds_data.present?
          process_odds(game, odds_data, results)
        else
          results[:odds_skipped] += 1
        end
      else
        results[:odds_skipped] += 1
      end

      # Process scores for in-progress and final games
      if game_state == "in" || (game_state == "post" && status_type["name"] == "STATUS_FINAL")
        process_scores(game, home_competitor, away_competitor, game_state, results)
      end
    end

    def mark_stale_games(espn_event_ids)
      Game.where(league: @league, game_date: @date)
          .where(game_date: Date.current..)
          .where("external_id NOT IN (?) OR external_id IS NULL", espn_event_ids)
          .where(is_stale: false)
          .update_all(is_stale: true)
    end

    def find_or_create_team(espn_abbreviation, espn_team_data, results)
      return nil unless espn_abbreviation

      # First: TeamIdentifier lookup for ESPN (handles overrides)
      identifier = TeamIdentifier.find_by(
        data_source: @espn_source,
        league: @league,
        external_code: espn_abbreviation
      )
      return identifier.team if identifier

      # Second: direct match on team code in this league
      team = @league.teams.find_by(code: espn_abbreviation)
      return team if team

      # Third: Create new team
      # Check for code collision with ANY existing team
      base_code = espn_abbreviation
      code = if Team.exists?(code: base_code)
        "#{base_code}_#{@league_code.upcase}"
      else
        base_code
      end

      team = Team.create!(
        code: code,
        location_name: espn_team_data["location"] || espn_abbreviation,
        nickname: espn_team_data["name"] || "Unknown",
        active: true
      )

      # Associate with league
      team.leagues << @league

      # Create ESPN identifier
      TeamIdentifier.create!(
        team: team,
        data_source: @espn_source,
        league: @league,
        external_code: espn_abbreviation
      )

      results[:teams_created] += 1
      puts "  ⚠️  New team created: #{code} (ESPN: #{espn_abbreviation}) - needs external mappings"

      team
    end

    def map_status(status_type)
      case status_type["name"]
      when "STATUS_SCHEDULED"
        "scheduled"
      when "STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD"
        "in_progress"
      when "STATUS_FINAL"
        "final"
      when "STATUS_POSTPONED"
        "postponed"
      when "STATUS_DELAYED"
        "delayed"
      when "STATUS_CANCELED", "STATUS_CANCELLED"
        "cancelled"
      else
        "scheduled"
      end
    end

    def process_odds(game, odds_data, results)
      # Check if we have any usable odds (spread or total)
      has_spread = odds_data["spread"].present? || odds_data.dig("pointSpread", "home", "close", "line").present?
      has_total = odds_data["overUnder"].present? || odds_data.dig("total", "over", "close", "line").present?
      
      unless has_spread || has_total
        results[:odds_skipped] += 1
        return
      end

      # Current favorite (for spread)
      home_is_favorite_now = odds_data.dig("homeTeamOdds", "favorite") == true
      favorite_team_now = home_is_favorite_now ? game.home_team : game.away_team

      # Favorite at open
      home_is_favorite_at_open = odds_data.dig("homeTeamOdds", "favoriteAtOpen") == true
      favorite_team_at_open = home_is_favorite_at_open ? game.home_team : game.away_team

      existing_odds = game.game_odds.where(data_source: @espn_source).order(fetched_at: :desc)
      has_opening = existing_odds.exists?(is_opening: true)

      # Opening odds - use favoriteAtOpen
      unless has_opening
        opening_odds = build_odds_from_espn(game, odds_data, favorite_team_at_open, home_is_favorite_at_open, use_open: true)
        if opening_odds
          opening_odds[:is_opening] = true
          GameOdds.create!(opening_odds)
          results[:odds_created] += 1
          puts "  Opening odds created: spread=#{opening_odds[:spread_value]}, total=#{opening_odds[:total_line]}"
        end
      end

      # Current odds - use current favorite
      current_odds = build_odds_from_espn(game, odds_data, favorite_team_now, home_is_favorite_now, use_open: false)
      return unless current_odds

      most_recent_current = existing_odds.where(is_opening: false).first

      if most_recent_current.nil? || odds_changed?(most_recent_current, current_odds)
        current_odds[:is_opening] = false
        GameOdds.create!(current_odds)
        results[:odds_created] += 1
        puts "  Current odds created: spread=#{current_odds[:spread_value]}, total=#{current_odds[:total_line]}"
      else
        results[:odds_skipped] += 1
      end
    end

    def build_odds_from_espn(game, odds_data, favorite_team, home_is_favorite, use_open:)
      key = use_open ? "open" : "close"

      # Helper to parse odds value
      parse_odds = ->(val) {
        return nil if val.nil? || val == "OFF"
        return 100 if val == "EVEN"
        val.to_i
      }

      spread_value = if use_open
        if home_is_favorite
          odds_data.dig("pointSpread", "home", "open", "line")&.to_f
        else
          odds_data.dig("pointSpread", "away", "open", "line")&.to_f
        end
      else
        if home_is_favorite
          odds_data.dig("pointSpread", "home", "close", "line")&.to_f
        else
          odds_data.dig("pointSpread", "away", "close", "line")&.to_f
        end
      end

      total_line = if use_open
        line_str = odds_data.dig("total", "over", "open", "line")
        line_str&.gsub(/[ou]/, '')&.to_f
      else
        odds_data["overUnder"]
      end

      return nil unless spread_value || total_line

      # Moneyline
      home_ml = parse_odds.call(odds_data.dig("moneyline", "home", key, "odds"))
      away_ml = parse_odds.call(odds_data.dig("moneyline", "away", key, "odds"))
      
      favorite_ml = home_is_favorite ? home_ml : away_ml
      underdog_ml = home_is_favorite ? away_ml : home_ml

      # Spread odds
      home_spread_odds = parse_odds.call(odds_data.dig("pointSpread", "home", key, "odds"))
      away_spread_odds = parse_odds.call(odds_data.dig("pointSpread", "away", key, "odds"))
      favorite_spread_odds = home_is_favorite ? home_spread_odds : away_spread_odds
      underdog_spread_odds = home_is_favorite ? away_spread_odds : home_spread_odds

      # Total odds
      over_odds = parse_odds.call(odds_data.dig("total", "over", key, "odds"))
      under_odds = parse_odds.call(odds_data.dig("total", "under", key, "odds"))

      {
        game: game,
        data_source: @espn_source,
        fetched_at: Time.current,
        spread_favorite_team: favorite_team,
        spread_value: spread_value,
        spread_favorite_odds: favorite_spread_odds,
        spread_underdog_odds: underdog_spread_odds,
        total_line: total_line,
        over_odds: over_odds,
        under_odds: under_odds,
        moneyline_favorite_team: favorite_team,
        moneyline_favorite_odds: favorite_ml,
        moneyline_underdog_odds: underdog_ml
      }
    end

    def odds_changed?(existing, new_odds)
      existing.spread_value != new_odds[:spread_value] ||
        existing.spread_favorite_odds != new_odds[:spread_favorite_odds] ||
        existing.spread_underdog_odds != new_odds[:spread_underdog_odds] ||
        existing.total_line != new_odds[:total_line] ||
        existing.over_odds != new_odds[:over_odds] ||
        existing.under_odds != new_odds[:under_odds] ||
        existing.moneyline_favorite_odds != new_odds[:moneyline_favorite_odds] ||
        existing.moneyline_underdog_odds != new_odds[:moneyline_underdog_odds]
    end

    def process_scores(game, home_competitor, away_competitor, game_state, results)
      home_score = home_competitor["score"]&.to_i
      away_score = away_competitor["score"]&.to_i

      return unless home_score && away_score

      is_final = game_state == "post"
      
      result = GameResult.find_or_initialize_by(game: game)
      
      scores_changed = result.new_record? || result.home_score != home_score || result.away_score != away_score
      final_changed = is_final && !result.final?

      if scores_changed || final_changed
        was_new = result.new_record?
        
        result.assign_attributes(
          home_score: home_score,
          away_score: away_score,
          final: is_final
        )
        result.save!
        
        results[:results_created] += 1 if was_new
        
        status_label = is_final ? "Final" : "Live"
        puts "  #{status_label}: #{game.away_team.code} #{away_score} - #{game.home_team.code} #{home_score}"
      end

      game.final! if is_final && !game.final?
    end
  end
end