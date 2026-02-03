# NCAA Men's Basketball Data Processor
#
# Processes raw Barthag data files and produces:
# 1. Team-level season summary (ncaam_team_data_final.csv)
# 2. Game-level data with rolling averages (base_model_game_data_with_rolling.csv)
#
# Usage:
#   # Single season (default - current season only)
#   service = Ncaam::DataProcessorService.new
#   service.call
#
#   # Multiple seasons for training data
#   service = Ncaam::DataProcessorService.new(
#     seasons: ['24_25', '25_26']  # Add '23_24' when ready
#   )
#   service.call
#
require 'csv'
require 'date'

module Ncaam
  class DataProcessorService
    HOME_ADVANTAGE = 3

    attr_reader :seasons, :team_stats_path, :team_table_path, :output_dir,
                :games_output_name, :team_output_name

    def initialize(seasons: ['25_26'], output_dir: nil)
      data_dir = Rails.root.join('db', 'data', 'ncaam')
      
      # Support single season string or array of seasons
      @seasons = Array(seasons)
      
      @team_stats_path = data_dir.join('raw', 'ncaam_barthag_team_stats_raw.csv')
      @team_table_path = data_dir.join('raw', 'ncaam_barthag_teams_table_raw.csv')
      @output_dir = output_dir || data_dir.join('processed')
      @games_output_name = 'base_model_game_data_with_rolling.csv'
      @team_output_name = 'ncaam_team_data_final.csv'
      @data_dir = data_dir
      
      @team_stats = nil
      @team_table = nil
      @rank_lookup = {}  # Team code -> Barttorvik rank
    end

    def call
      puts '=' * 60
      puts 'Loading source files...'
      puts '=' * 60

      @team_stats = load_team_stats
      puts "  Loaded #{@team_stats.length} teams from team stats"

      @team_table = load_team_table
      puts "  Loaded #{@team_table.length} teams from team table"

      # Build rank lookup from team table
      build_rank_lookup
      puts "  Built rank lookup for #{@rank_lookup.length} teams"

      puts
      puts '=' * 60
      puts 'Processing game data...'
      puts '=' * 60

      # Process all seasons
      all_games = process_all_seasons
      puts "  Total games across all seasons: #{all_games.length}"

      games_with_rolling = calculate_rolling_averages(all_games)
      games_with_sos = add_sos_to_games(games_with_rolling)
      
      # Add home advantage constant
      games_with_sos.each { |g| g[:home_advantage] = HOME_ADVANTAGE }

      puts
      puts '=' * 60
      puts 'Calculating quad records...'
      puts '=' * 60

      # Calculate quad records for each team based on game history
      quad_records = calculate_quad_records(all_games)
      puts "  Calculated quad records for #{quad_records.length} teams"

      puts
      puts '=' * 60
      puts 'Creating team summary...'
      puts '=' * 60

      team_summary = create_team_summary(quad_records)
      puts "  Created summary for #{team_summary.length} teams"

      puts
      puts '=' * 60
      puts 'Saving output files...'
      puts '=' * 60

      save_games_csv(games_with_sos)
      save_team_summary_csv(team_summary)

      puts
      puts '=' * 60
      puts 'Processing complete!'
      puts '=' * 60

      { games: games_with_sos, team_summary: team_summary }
    end

    private

    # =========================================================================
    # Rank Lookup
    # =========================================================================

    def build_rank_lookup
      @team_table.each do |team|
        code = team['Team_Code']
        rank = to_i_safe(team['Team ID'])
        @rank_lookup[code] = rank if code && rank
      end
    end

    # =========================================================================
    # Quad Calculations
    # =========================================================================

    def get_quad(opp_rank, location)
      # Determine quad level based on opponent rank and game location
      # location: 'home', 'away', or 'neutral'
      return nil unless opp_rank

      if location == 'home'
        if opp_rank <= 30 then 1
        elsif opp_rank <= 75 then 2
        elsif opp_rank <= 160 then 3
        else 4
        end
      elsif location == 'neutral'
        if opp_rank <= 50 then 1
        elsif opp_rank <= 100 then 2
        elsif opp_rank <= 200 then 3
        else 4
        end
      else # away
        if opp_rank <= 75 then 1
        elsif opp_rank <= 135 then 2
        elsif opp_rank <= 240 then 3
        else 4
        end
      end
    end

    def calculate_quad_records(games_data)
      # Initialize records for all teams
      quad_records = Hash.new do |h, k|
        h[k] = {
          q1_wins: 0, q1_losses: 0,
          q2_wins: 0, q2_losses: 0,
          q3_wins: 0, q3_losses: 0,
          q4_wins: 0, q4_losses: 0
        }
      end

      games_data.each do |game|
        away_team = game[:away_team]
        home_team = game[:home_team]
        away_score = game[:away_score]
        home_score = game[:home_score]
        venue = game[:venue]

        # Skip if missing data
        next unless away_team && home_team && away_score && home_score

        away_rank = @rank_lookup[away_team]
        home_rank = @rank_lookup[home_team]

        # Determine locations
        if venue == 'N'
          away_location = 'neutral'
          home_location = 'neutral'
        else
          away_location = 'away'
          home_location = 'home'
        end

        # Away team's perspective
        away_quad = get_quad(home_rank, away_location)
        if away_quad
          if away_score > home_score
            quad_records[away_team][:"q#{away_quad}_wins"] += 1
          else
            quad_records[away_team][:"q#{away_quad}_losses"] += 1
          end
        end

        # Home team's perspective
        home_quad = get_quad(away_rank, home_location)
        if home_quad
          if home_score > away_score
            quad_records[home_team][:"q#{home_quad}_wins"] += 1
          else
            quad_records[home_team][:"q#{home_quad}_losses"] += 1
          end
        end
      end

      # Calculate derived quad metrics for each team
      quad_records.each do |team, record|
        # Quality wins (Q1 + Q2)
        record[:q1_q2_wins] = record[:q1_wins] + record[:q2_wins]
        record[:q1_q2_games] = record[:q1_wins] + record[:q1_losses] + record[:q2_wins] + record[:q2_losses]
        record[:q1_q2_win_pct] = record[:q1_q2_games] > 0 ? 
          (record[:q1_q2_wins].to_f / record[:q1_q2_games] * 100).round(1) : 0

        # Bad losses (Q3 + Q4)
        record[:q3_q4_losses] = record[:q3_losses] + record[:q4_losses]

        # Quality score: Q1 wins - Q4 losses
        record[:quality_score] = record[:q1_wins] - record[:q4_losses]

        # Weighted quality: emphasizes Q1 wins and penalizes Q4 losses
        # Q1 wins = +3, Q2 wins = +2, Q3 wins = +1, Q4 wins = 0
        # Q3 losses = -1, Q4 losses = -3
        record[:weighted_quality] = (
          3 * record[:q1_wins] +
          2 * record[:q2_wins] +
          1 * record[:q3_wins] +
          0 * record[:q4_wins] -
          1 * record[:q3_losses] -
          3 * record[:q4_losses]
        )
      end

      quad_records
    end

    # =========================================================================
    # Multi-Season Processing
    # =========================================================================

    def process_all_seasons
      all_games = []
      game_id_offset = 0

      # Process seasons in chronological order (oldest first)
      sorted_seasons = @seasons.sort

      sorted_seasons.each do |season|
        games_path = @data_dir.join('raw', "ncaam_#{season}_games_raw.csv")
        
        unless File.exist?(games_path)
          puts "  Skipping season #{season} - file not found: #{games_path}"
          next
        end

        puts "  Processing season #{season}..."
        
        # Load raw game rows for this season
        raw_games = load_games(games_path)
        puts "    Loaded #{raw_games.length} game rows"

        # Merge into single row per game
        merged_games = merge_games(raw_games)
        puts "    Merged into #{merged_games.length} games"

        # Add season identifier and offset game IDs
        merged_games.each_with_index do |game, idx|
          game[:season] = season
          game[:id] = game_id_offset + idx + 1
        end

        game_id_offset += merged_games.length
        all_games.concat(merged_games)
      end

      # Final sort by date across all seasons
      all_games.sort_by! { |g| g[:date] || '0000-00-00' }
      
      # Re-assign IDs after final sort
      all_games.each_with_index { |g, idx| g[:id] = idx + 1 }

      all_games
    end

    # =========================================================================
    # Team Code Mapping
    # =========================================================================

    def team_name_to_code(team_name, league_code = 'ncaam')
      return nil if team_name.nil? || team_name.to_s.strip.empty?
      
      identifier = TeamIdentifier.joins(:data_source, :league)
                                .where(data_sources: { code: 'barttorvik' }, 
                                        leagues: { code: league_code },
                                        external_code: team_name.strip)
                                .first
      
      identifier&.team&.code
    end

    # =========================================================================
    # Utility Functions
    # =========================================================================

    def parse_value_rank_cell(cell)
      return [nil, nil] if cell.nil? || cell.to_s.strip.empty?
      
      cell_str = cell.to_s.strip
      if cell_str.include?("\n")
        parts = cell_str.split("\n")
        begin
          value = parts[0].strip.to_f
          rank = parts[1].strip.to_i
          [value, rank]
        rescue
          [nil, nil]
        end
      else
        begin
          [cell_str.to_f, nil]
        rescue
          [nil, nil]
        end
      end
    end

    def parse_result(result_str)
      return [nil, nil] if result_str.nil?
      
      match = result_str.to_s.strip.match(/([WL]),\s*(\d+)-(\d+)/)
      return [nil, nil] unless match
      
      outcome = match[1]
      score1 = match[2].to_i
      score2 = match[3].to_i
      
      if outcome == 'W'
        [score1, score2]  # Team won, so team has higher score
      else
        [score2, score1]  # Team lost, so team has lower score
      end
    end

    def parse_date(date_str)
      return nil if date_str.nil?
      
      date_str = date_str.to_s.strip
      
      # Try different formats
      ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d'].each do |fmt|
        begin
          return Date.strptime(date_str, fmt).strftime('%Y-%m-%d')
        rescue ArgumentError
          next
        end
      end
      
      date_str  # Return as-is if parsing fails
    end

    def header_row?(row)
      first_val = row[0].to_s.strip.downcase
      second_val = row[1].to_s.strip.downcase rescue ''
      
      %w[rk rank # ''].include?(first_val) || %w[team ''].include?(second_val)
    end

    def to_f_safe(val)
      return nil if val.nil? || val.to_s.strip.empty?
      val.to_s.strip.to_f
    end

    def to_i_safe(val)
      return nil if val.nil? || val.to_s.strip.empty?
      val.to_s.strip.to_i
    end

    # =========================================================================
    # Data Loading Functions
    # =========================================================================

    def load_team_stats
      puts "Loading team stats from: #{team_stats_path}"
      
      rows = CSV.read(team_stats_path)
      
      # Skip the first header row (parent header)
      rows = rows[2..]  # Skip both header rows
      
      # Remove duplicate header rows
      rows = rows.reject { |row| header_row?(row) }
      
      stat_mappings = [
        ['AdjEff_Off', 'Adj. Off. Eff', 'Adj. Off. Eff Rank'],
        ['AdjEff_Def', 'Adj. Def. Eff', 'Adj. Def. Eff Rank'],
        ['eFG_Off', 'Eff. FG% Off', 'Eff. FG% Off Rank'],
        ['eFG_Def', 'Eff. FG% Def', 'Eff. FG% Def Rank'],
        ['TO_Off', 'Turnover% Off', 'Turnover% Off Rank'],
        ['TO_Def', 'Turnover% Def', 'Turnover% Def Rank'],
        ['OReb_Off', 'Off. Reb%', 'Off. Reb% Rank'],
        ['OReb_Def', 'Def. Reb%', 'Def. Reb% Rank'],
        ['FTRate_Off', 'FT Rate Off', 'FT Rate Off Rank'],
        ['FTRate_Def', 'FT Rate Def', 'FT Rate Def Rank'],
        ['FT_Off', 'FT% Off', 'FT% Off Rank'],
        ['FT_Def', 'FT% Def', 'FT% Def Rank'],
        ['2P_Off', '2P% Off', '2P% Off Rank'],
        ['2P_Def', '2P% Def', '2P% Def Rank'],
        ['3P_Off', '3P% Off', '3P% Off Rank'],
        ['3P_Def', '3P% Def', '3P% Def Rank'],
      ]
      
      parsed_data = rows.map do |row|
        team_data = {
          'Team' => row[1],
          'Team_Code' => team_name_to_code(row[1]),
          'Conference' => row[2]
        }
        
        stat_mappings.each_with_index do |(_, val_name, rank_name), idx|
          col_idx = idx + 3  # Start after Rk, Team, Conf
          next if col_idx >= row.length
          
          value, rank = parse_value_rank_cell(row[col_idx])
          team_data[val_name] = value
          team_data[rank_name] = rank
        end
        
        team_data
      end
      
      parsed_data
    end

    def load_team_table
      puts "Loading team table from: #{team_table_path}"
      
      rows = CSV.read(team_table_path, headers: true)
      
      column_mapping = {
        rows.headers[0] => 'Team ID',
        'Team' => 'Team',
        'Adj OE' => 'Adj. Off. Eff',
        'Adj DE' => 'Adj. Def. Eff',
        'Barthag' => 'Barthag',
        'Wins' => 'Wins',
        'Games' => 'Games',
        'eFG' => 'Eff. FG% Off',
        'eFG D.' => 'Eff. FG% Def',
        'FT Rate' => 'FT Rate Off',
        'FT Rate D' => 'FT Rate Def',
        'TOV%' => 'Turnover% Off',
        'TOV% D' => 'Turnover% Def',
        'O Reb%' => 'Off. Reb%',
        'Op OReb%' => 'Def. Reb%',
        'Raw T' => 'Raw Tempo',
        '2P %' => '2P% Off',
        '2P % D.' => '2P% Def',
        '3P %' => '3P% Off',
        '3P % D.' => '3P% Def',
        'Blk %' => 'Block% Def',
        'Blked %' => 'Block% Off',
        'Ast %' => 'Assist% Off',
        'Op Ast %' => 'Assist% Def',
        '3P Rate' => '3P Rate Off',
        '3P Rate D' => '3P Rate Def',
        'Adj. T' => 'Adj. Tempo',
        'Avg Hgt.' => 'Avg Height',
        'Eff. Hgt.' => 'Eff. Height',
        'Exp.' => 'Experience',
        'Talent' => 'Talent',
        'FT%' => 'FT% Off',
        'Op. FT%' => 'FT% Def',
        'PPP Off.' => 'PPP Off',
        'PPP Def.' => 'PPP Def',
        'Elite SOS' => 'Elite SOS'
      }
      
      rows.map do |row|
        team_data = {}
        
        row.each do |header, value|
          new_key = column_mapping[header] || header
          team_data[new_key] = value
        end
        
        team_data['Team_Code'] = team_name_to_code(team_data['Team'])
        team_data
      end
    end

    def load_games(games_path)
      puts "  Loading games from: #{games_path}"
      
      rows = CSV.read(games_path)
      
      # Skip the first header row (parent header for Offense/Defense)
      rows = rows[2..]
      
      # Remove duplicate header rows
      rows = rows.reject { |row| header_row?(row) }
      
      expected_cols = %w[
        row_num Rk Date Type Team Conf Opp Venue Result
        Adj_O Adj_D T
        OEFF OeFG OTO OReb OFTR
        DEFF DeFG DTO DReb DFTR
        G_Sc Plus_Minus
      ]
      
      numeric_cols = %w[Adj_O Adj_D T OEFF OeFG OTO OReb OFTR DEFF DeFG DTO DReb DFTR G_Sc Plus_Minus]
      
      rows.map do |row|
        game_data = {}
        expected_cols.each_with_index do |col, idx|
          next if idx >= row.length
          
          val = row[idx]
          if numeric_cols.include?(col)
            game_data[col] = to_f_safe(val)
          elsif col == 'Date'
            game_data[col] = parse_date(val)
          else
            game_data[col] = val
          end
        end
        
        game_data
      end
    end

    # =========================================================================
    # Data Processing Functions
    # =========================================================================

    def merge_games(games_data)
      # Sort by date
      sorted_games = games_data.sort_by { |g| g['Date'] || '0000-00-00' }
      
      # Create game key for matching
      make_game_key = ->(row) {
        teams = [row['Team'].to_s, row['Opp'].to_s].sort
        "#{row['Date']}_#{teams[0]}_#{teams[1]}"
      }
      
      # Group by game key
      grouped = sorted_games.group_by { |row| make_game_key.call(row) }
      
      merged_games = []
      
      grouped.each do |game_key, game_rows|
        next if game_rows.length != 2
        
        row1, row2 = game_rows
        
        # Determine away/home based on Venue
        venue1 = row1['Venue']
        venue2 = row2['Venue']
        
        if venue1 == 'H'
          home_row, away_row = row1, row2
          venue = ''
        elsif venue2 == 'H'
          home_row, away_row = row2, row1
          venue = ''
        elsif venue1 == 'A'
          away_row, home_row = row1, row2
          venue = ''
        elsif venue2 == 'A'
          away_row, home_row = row2, row1
          venue = ''
        else
          # Both neutral - arbitrary assignment
          away_row, home_row = row1, row2
          venue = 'N'
        end
        
        # Parse scores from result string
        away_team_score, away_opp_score = parse_result(away_row['Result'])
        
        merged_game = {
          date: row1['Date'],
          away_team: team_name_to_code(away_row['Team']),
          away_team_name: away_row['Team'],
          away_conf: away_row['Conf'],
          home_team: team_name_to_code(home_row['Team']),
          home_team_name: home_row['Team'],
          home_conf: home_row['Conf'],
          venue: venue,
          away_score: away_team_score,
          home_score: away_opp_score,
          
          # Interleaved away/home stats
          away_AdjO: away_row['Adj_O'],
          home_AdjO: home_row['Adj_O'],
          away_AdjD: away_row['Adj_D'],
          home_AdjD: home_row['Adj_D'],
          away_T: away_row['T'],
          home_T: home_row['T'],
          away_OEFF: away_row['OEFF'],
          home_OEFF: home_row['OEFF'],
          'away_OeFG%': away_row['OeFG'],
          'home_OeFG%': home_row['OeFG'],
          'away_OTO%': away_row['OTO'],
          'home_OTO%': home_row['OTO'],
          'away_OReb%': away_row['OReb'],
          'home_OReb%': home_row['OReb'],
          away_OFTR: away_row['OFTR'],
          home_OFTR: home_row['OFTR'],
          away_DEFF: away_row['DEFF'],
          home_DEFF: home_row['DEFF'],
          'away_DeFG%': away_row['DeFG'],
          'home_DeFG%': home_row['DeFG'],
          'away_DTO%': away_row['DTO'],
          'home_DTO%': home_row['DTO'],
          'away_DReb%': away_row['DReb'],
          'home_DReb%': home_row['DReb'],
          away_DFTR: away_row['DFTR'],
          home_DFTR: home_row['DFTR'],
          away_g_score: away_row['G_Sc'],
          home_g_score: home_row['G_Sc'],
        }
        
        merged_games << merged_game
      end
      
      # Sort by date (IDs assigned later in process_all_seasons)
      merged_games.sort_by! { |g| g[:date] || '0000-00-00' }
      
      merged_games
    end

    def calculate_rolling_averages(games_data)
      puts 'Calculating rolling averages...'
      
      # Build team game history with all stats we need to roll
      team_games = []
      
      games_data.each do |game|
        # Away team's game
        team_games << {
          date: game[:date],
          season: game[:season],
          team: game[:away_team],
          game_id: game[:id],
          AdjO: game[:away_AdjO],
          AdjD: game[:away_AdjD],
          T: game[:away_T],
          eFG_off: game[:"away_OeFG%"],
          TOV_off: game[:"away_OTO%"],
          OReb: game[:"away_OReb%"],
          FTR_off: game[:away_OFTR],
          eFG_def: game[:"away_DeFG%"],
          TOV_def: game[:"away_DTO%"],
          DReb: game[:"away_DReb%"],
          FTR_def: game[:away_DFTR],
          g_score: game[:away_g_score]
        }
        
        # Home team's game
        team_games << {
          date: game[:date],
          season: game[:season],
          team: game[:home_team],
          game_id: game[:id],
          AdjO: game[:home_AdjO],
          AdjD: game[:home_AdjD],
          T: game[:home_T],
          eFG_off: game[:"home_OeFG%"],
          TOV_off: game[:"home_OTO%"],
          OReb: game[:"home_OReb%"],
          FTR_off: game[:home_OFTR],
          eFG_def: game[:"home_DeFG%"],
          TOV_def: game[:"home_DTO%"],
          DReb: game[:"home_DReb%"],
          FTR_def: game[:home_DFTR],
          g_score: game[:home_g_score]
        }
      end
      
      # Sort by team, then by date
      team_games.sort_by! { |g| [g[:team] || '', g[:date] || '0000-00-00'] }
      
      # Group by team
      by_team = team_games.group_by { |g| g[:team] }
      
      # Calculate rolling averages
      rolling_stats = {}
      
      by_team.each do |team, games|
        next unless team
        
        prev_date = nil
        
        games.each_with_index do |game, idx|
          game_id = game[:game_id]
          rolling_stats[game_id] ||= {}
          
          # Days rest calculation
          if prev_date && game[:date]
            begin
              current = Date.parse(game[:date])
              previous = Date.parse(prev_date)
              days_rest = (current - previous).to_i
            rescue
              days_rest = 7
            end
          else
            days_rest = 7
          end
          prev_date = game[:date]
          
          # Get previous N games (excluding current)
          prev_5 = games[[0, idx - 5].max...idx]
          prev_10 = games[[0, idx - 10].max...idx]
          
          # Calculate averages
          stats_to_roll = [:AdjO, :AdjD, :T, :eFG_off, :TOV_off, :OReb, :FTR_off, 
                          :eFG_def, :TOV_def, :DReb, :FTR_def, :g_score]
          
          rolling_5 = {}
          rolling_10 = {}
          
          stats_to_roll.each do |stat|
            vals_5 = prev_5.map { |g| g[stat] }.compact
            vals_10 = prev_10.map { |g| g[stat] }.compact
            
            rolling_5[stat] = vals_5.any? ? vals_5.sum / vals_5.length : nil
            rolling_10[stat] = vals_10.any? ? vals_10.sum / vals_10.length : nil
          end
          
          # Find the original game to determine away/home prefix
          original_game = games_data.find { |g| g[:id] == game_id }
          next unless original_game
          
          prefix = original_game[:away_team] == team ? 'away' : 'home'
          
          rolling_stats[game_id][:"#{prefix}_AdjO_rolling_5"] = rolling_5[:AdjO]&.round(2)
          rolling_stats[game_id][:"#{prefix}_AdjO_rolling_10"] = rolling_10[:AdjO]&.round(2)
          rolling_stats[game_id][:"#{prefix}_AdjD_rolling_5"] = rolling_5[:AdjD]&.round(2)
          rolling_stats[game_id][:"#{prefix}_AdjD_rolling_10"] = rolling_10[:AdjD]&.round(2)
          rolling_stats[game_id][:"#{prefix}_T_rolling_5"] = rolling_5[:T]&.round(2)
          rolling_stats[game_id][:"#{prefix}_T_rolling_10"] = rolling_10[:T]&.round(2)
          rolling_stats[game_id][:"#{prefix}_eFG_off_rolling_5"] = rolling_5[:eFG_off]&.round(2)
          rolling_stats[game_id][:"#{prefix}_eFG_off_rolling_10"] = rolling_10[:eFG_off]&.round(2)
          rolling_stats[game_id][:"#{prefix}_TOV_off_rolling_5"] = rolling_5[:TOV_off]&.round(2)
          rolling_stats[game_id][:"#{prefix}_TOV_off_rolling_10"] = rolling_10[:TOV_off]&.round(2)
          rolling_stats[game_id][:"#{prefix}_OReb_rolling_5"] = rolling_5[:OReb]&.round(2)
          rolling_stats[game_id][:"#{prefix}_OReb_rolling_10"] = rolling_10[:OReb]&.round(2)
          rolling_stats[game_id][:"#{prefix}_FTR_off_rolling_5"] = rolling_5[:FTR_off]&.round(2)
          rolling_stats[game_id][:"#{prefix}_FTR_off_rolling_10"] = rolling_10[:FTR_off]&.round(2)
          rolling_stats[game_id][:"#{prefix}_eFG_def_rolling_5"] = rolling_5[:eFG_def]&.round(2)
          rolling_stats[game_id][:"#{prefix}_eFG_def_rolling_10"] = rolling_10[:eFG_def]&.round(2)
          rolling_stats[game_id][:"#{prefix}_TOV_def_rolling_5"] = rolling_5[:TOV_def]&.round(2)
          rolling_stats[game_id][:"#{prefix}_TOV_def_rolling_10"] = rolling_10[:TOV_def]&.round(2)
          rolling_stats[game_id][:"#{prefix}_DReb_rolling_5"] = rolling_5[:DReb]&.round(2)
          rolling_stats[game_id][:"#{prefix}_DReb_rolling_10"] = rolling_10[:DReb]&.round(2)
          rolling_stats[game_id][:"#{prefix}_FTR_def_rolling_5"] = rolling_5[:FTR_def]&.round(2)
          rolling_stats[game_id][:"#{prefix}_FTR_def_rolling_10"] = rolling_10[:FTR_def]&.round(2)
          rolling_stats[game_id][:"#{prefix}_g_score_rolling_5"] = rolling_5[:g_score]&.round(2)
          rolling_stats[game_id][:"#{prefix}_g_score_rolling_10"] = rolling_10[:g_score]&.round(2)
          rolling_stats[game_id][:"#{prefix}_days_rest"] = days_rest
        end
      end
      
      # Merge rolling stats back into games
      games_data.each do |game|
        stats = rolling_stats[game[:id]] || {}
        game.merge!(stats)
      end
      
      games_data
    end

    def add_sos_to_games(games_data)
      puts 'Adding SOS to games...'
      
      sos_lookup = {}
      @team_table.each do |team|
        code = team['Team_Code']
        sos_lookup[code] = to_f_safe(team['Elite SOS']) if code
      end
      
      games_data.each do |game|
        game[:away_sos] = sos_lookup[game[:away_team]]
        game[:home_sos] = sos_lookup[game[:home_team]]
      end
      
      games_data
    end

    def create_team_summary(quad_records = {})
      puts 'Creating team summary...'
      
      stats_by_team = {}
      @team_stats.each do |team|
        stats_by_team[team['Team']] = team
      end
      
      merged = @team_table.map do |team|
        team_name = team['Team']
        stats = stats_by_team[team_name] || {}
        
        result = team.merge(stats) do |key, old_val, new_val|
          key.include?('Rank') ? new_val : old_val
        end
        
        result['Conference'] ||= stats['Conference']
        
        # Add quad records if available
        team_code = result['Team_Code']
        if team_code && quad_records[team_code]
          qr = quad_records[team_code]
          result['Q1 Wins'] = qr[:q1_wins]
          result['Q1 Losses'] = qr[:q1_losses]
          result['Q2 Wins'] = qr[:q2_wins]
          result['Q2 Losses'] = qr[:q2_losses]
          result['Q3 Wins'] = qr[:q3_wins]
          result['Q3 Losses'] = qr[:q3_losses]
          result['Q4 Wins'] = qr[:q4_wins]
          result['Q4 Losses'] = qr[:q4_losses]
          result['Q1-Q2 Wins'] = qr[:q1_q2_wins]
          result['Q1-Q2 Win Pct'] = qr[:q1_q2_win_pct]
          result['Q3-Q4 Losses'] = qr[:q3_q4_losses]
          result['Quality Score'] = qr[:quality_score]
          result['Weighted Quality'] = qr[:weighted_quality]
        end
        
        result
      end
      
      # Calculate any missing ranks
      rank_calcs = [
        ['Block% Def', 'Block% Def Rank', true],
        ['Block% Off', 'Block% Off Rank', false],
        ['Assist% Off', 'Assist% Off Rank', true],
        ['Assist% Def', 'Assist% Def Rank', false],
        ['3P Rate Off', '3P Rate Off Rank', true],
        ['3P Rate Def', '3P Rate Def Rank', false],
      ]
      
      rank_calcs.each do |col, rank_col, higher_is_better|
        values = merged.map { |t| [t, to_f_safe(t[col])] }.select { |_, v| v }
        
        sorted = if higher_is_better
          values.sort_by { |_, v| -v }
        else
          values.sort_by { |_, v| v }
        end
        
        sorted.each_with_index do |(team, _), idx|
          team[rank_col] = idx + 1
        end
      end
      
      merged
    end

    # =========================================================================
    # Output Functions
    # =========================================================================

    def save_games_csv(games_data)
      output_path = File.join(output_dir, games_output_name)
      
      column_order = %i[
        id season date away_team away_conf home_team home_conf venue
        away_score home_score
        away_AdjO home_AdjO away_AdjD home_AdjD away_T home_T
        away_OEFF home_OEFF away_OeFG% home_OeFG% away_OTO% home_OTO%
        away_OReb% home_OReb% away_OFTR home_OFTR
        away_DEFF home_DEFF away_DeFG% home_DeFG% away_DTO% home_DTO%
        away_DReb% home_DReb% away_DFTR home_DFTR
        away_g_score home_g_score
        away_AdjO_rolling_5 away_AdjO_rolling_10 home_AdjO_rolling_5 home_AdjO_rolling_10
        away_AdjD_rolling_5 away_AdjD_rolling_10 home_AdjD_rolling_5 home_AdjD_rolling_10
        away_T_rolling_5 away_T_rolling_10 home_T_rolling_5 home_T_rolling_10
        away_eFG_off_rolling_5 away_eFG_off_rolling_10 home_eFG_off_rolling_5 home_eFG_off_rolling_10
        away_TOV_off_rolling_5 away_TOV_off_rolling_10 home_TOV_off_rolling_5 home_TOV_off_rolling_10
        away_OReb_rolling_5 away_OReb_rolling_10 home_OReb_rolling_5 home_OReb_rolling_10
        away_FTR_off_rolling_5 away_FTR_off_rolling_10 home_FTR_off_rolling_5 home_FTR_off_rolling_10
        away_eFG_def_rolling_5 away_eFG_def_rolling_10 home_eFG_def_rolling_5 home_eFG_def_rolling_10
        away_TOV_def_rolling_5 away_TOV_def_rolling_10 home_TOV_def_rolling_5 home_TOV_def_rolling_10
        away_DReb_rolling_5 away_DReb_rolling_10 home_DReb_rolling_5 home_DReb_rolling_10
        away_FTR_def_rolling_5 away_FTR_def_rolling_10 home_FTR_def_rolling_5 home_FTR_def_rolling_10
        away_g_score_rolling_5 away_g_score_rolling_10 home_g_score_rolling_5 home_g_score_rolling_10
        away_days_rest home_days_rest
        away_sos home_sos home_advantage
      ]
      
      headers = column_order.map(&:to_s)
      
      CSV.open(output_path, 'w', quote_char: '"', force_quotes: true) do |csv|
        csv << headers
        
        games_data.each do |game|
          row = column_order.map { |col| game[col] || game[col.to_s] }
          csv << row
        end
      end
      
      puts "  Saved games data to: #{output_path}"
      puts "  Total games: #{games_data.length}"
    end

    def save_team_summary_csv(team_data)
      output_path = File.join(output_dir, team_output_name)
      
      output_columns = [
        'Team ID', 'Team', 'Team_Code', 'Conference',
        'Adj. Off. Eff', 'Adj. Off. Eff Rank',
        'Adj. Def. Eff', 'Adj. Def. Eff Rank',
        'Barthag', 'Wins', 'Games',
        'Eff. FG% Off', 'Eff. FG% Off Rank',
        'Eff. FG% Def', 'Eff. FG% Def Rank',
        'FT Rate Off', 'FT Rate Off Rank',
        'FT Rate Def', 'FT Rate Def Rank',
        'Turnover% Off', 'Turnover% Off Rank',
        'Turnover% Def', 'Turnover% Def Rank',
        'Off. Reb%', 'Off. Reb% Rank',
        'Def. Reb%', 'Def. Reb% Rank',
        'Raw Tempo',
        '2P% Off', '2P% Off Rank',
        '2P% Def', '2P% Def Rank',
        '3P% Off', '3P% Off Rank',
        '3P% Def', '3P% Def Rank',
        'Block% Off', 'Block% Def',
        'Assist% Off', 'Assist% Def',
        '3P Rate Off', '3P Rate Def',
        'Adj. Tempo',
        'Avg Height', 'Eff. Height',
        'Experience', 'Talent',
        'FT% Off', 'FT% Off Rank',
        'FT% Def', 'FT% Def Rank',
        'PPP Off', 'PPP Def',
        'Elite SOS',
        'Block% Def Rank', 'Block% Off Rank',
        'Assist% Off Rank', 'Assist% Def Rank',
        '3P Rate Off Rank', '3P Rate Def Rank',
        # Quad records
        'Q1 Wins', 'Q1 Losses',
        'Q2 Wins', 'Q2 Losses',
        'Q3 Wins', 'Q3 Losses',
        'Q4 Wins', 'Q4 Losses',
        'Q1-Q2 Wins', 'Q1-Q2 Win Pct',
        'Q3-Q4 Losses', 'Quality Score', 'Weighted Quality'
      ]
      
      available_columns = output_columns.select do |col|
        team_data.any? { |t| t.key?(col) }
      end
      
      CSV.open(output_path, 'w') do |csv|
        csv << available_columns
        
        team_data.each do |team|
          row = available_columns.map do |col|
            val = team[col]
            if col.include?('Rank') || col == 'Team ID' || col.include?('Wins') || col.include?('Losses') || col == 'Quality Score' || col == 'Weighted Quality'
              val.to_i if val
            else
              val
            end
          end
          csv << row
        end
      end
      
      puts "  Saved team summary to: #{output_path}"
    end
  end
end