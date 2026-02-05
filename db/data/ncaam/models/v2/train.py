"""
NCAAM Model v2: Building models with derived features

This script:
1. Loads data and creates derived features (using same logic as correlations script)
2. Selects non-redundant feature set
3. Trains separate models for each target (score, total, spread, win)
4. Compares performance to baseline and Vegas
5. Outputs feature importance and error analysis

Usage:
    python ncaam_model_v2.py \
        --bart-games /path/to/base_model_game_data_with_rolling.csv \
        --bart-season /path/to/ncaam_team_data_final.csv \
        --espn-team /path/to/team_stats.csv \
        --espn-player /path/to/player_stats.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb
import argparse
import warnings
warnings.filterwarnings('ignore')

LEAGUE_AVG_TEMPO = 68.0
LEAGUE_AVG_EFFICIENCY = 100.0


# =============================================================================
# DATA LOADING AND PREPARATION (copied from correlations script)
# =============================================================================

def load_data(bart_games_path, bart_season_path, espn_team_path, espn_player_path):
    """Load all data sources."""
    print("Loading data...")
    
    bart_games = pd.read_csv(bart_games_path)
    bart_season = pd.read_csv(bart_season_path)
    espn_team = pd.read_csv(espn_team_path)
    espn_player = pd.read_csv(espn_player_path)
    
    print(f"  Barttorvik games: {len(bart_games)} rows")
    print(f"  Barttorvik season: {len(bart_season)} rows")
    print(f"  ESPN team box scores: {len(espn_team)} rows")
    print(f"  ESPN player box scores: {len(espn_player)} rows")
    
    return bart_games, bart_season, espn_team, espn_player


def prepare_season_data(bart_season):
    """Prepare season-level team data with consistent column naming.
    
    Actual columns from ncaam_team_data_final.csv:
    Team_Code, Adj. Off. Eff, Adj. Def. Eff, Eff. FG% Off, Turnover% Off, etc.
    """
    print("Preparing season data...")
    
    df = bart_season.copy()
    
    # Actual column names from ncaam_team_data_final.csv -> our internal names
    column_mapping = {
        'Team_Code': 'team_code',
        'Team': 'team_name',
        'Conference': 'conference',
        'Adj. Off. Eff': 'adjO',
        'Adj. Off. Eff Rank': 'adjO_rank',
        'Adj. Def. Eff': 'adjD',
        'Adj. Def. Eff Rank': 'adjD_rank',
        'Barthag': 'barthag',
        'Wins': 'wins',
        'Games': 'games',
        'Eff. FG% Off': 'efg_off',
        'Eff. FG% Off Rank': 'efg_off_rank',
        'Eff. FG% Def': 'efg_def',
        'Eff. FG% Def Rank': 'efg_def_rank',
        'FT Rate Off': 'ftr_off',
        'FT Rate Off Rank': 'ftr_off_rank',
        'FT Rate Def': 'ftr_def',
        'FT Rate Def Rank': 'ftr_def_rank',
        'Turnover% Off': 'tov_off',
        'Turnover% Off Rank': 'tov_off_rank',
        'Turnover% Def': 'tov_def',
        'Turnover% Def Rank': 'tov_def_rank',
        'Off. Reb%': 'oreb_pct',
        'Off. Reb% Rank': 'oreb_rank',
        'Def. Reb%': 'dreb_pct',
        'Def. Reb% Rank': 'dreb_rank',
        'Raw Tempo': 'tempo_raw',
        'Adj. Tempo': 'tempo_adj',
        '2P% Off': '2pt_off',
        '2P% Off Rank': '2pt_off_rank',
        '2P% Def': '2pt_def',
        '2P% Def Rank': '2pt_def_rank',
        '3P% Off': '3pt_off',
        '3P% Off Rank': '3pt_off_rank',
        '3P% Def': '3pt_def',
        '3P% Def Rank': '3pt_def_rank',
        'Block% Off': 'blk_off',
        'Block% Off Rank': 'blk_off_rank',
        'Block% Def': 'blk_def',
        'Block% Def Rank': 'blk_def_rank',
        'Assist% Off': 'ast_off',
        'Assist% Off Rank': 'ast_off_rank',
        'Assist% Def': 'ast_def',
        'Assist% Def Rank': 'ast_def_rank',
        '3P Rate Off': '3pt_rate_off',
        '3P Rate Off Rank': '3pt_rate_off_rank',
        '3P Rate Def': '3pt_rate_def',
        '3P Rate Def Rank': '3pt_rate_def_rank',
        'Avg Height': 'avg_height',
        'Eff. Height': 'eff_height',
        'Experience': 'experience',
        'Talent': 'talent',
        'FT% Off': 'ft_pct_off',
        'FT% Off Rank': 'ft_pct_off_rank',
        'FT% Def': 'ft_pct_def',
        'FT% Def Rank': 'ft_pct_def_rank',
        'PPP Off': 'ppp_off',
        'PPP Def': 'ppp_def',
        'Elite SOS': 'elite_sos',
        'Team ID': 'bart_rank',  # Barttorvik overall ranking (1 = best)
        # Quad records
        'Q1 Wins': 'q1_wins',
        'Q1 Losses': 'q1_losses',
        'Q2 Wins': 'q2_wins',
        'Q2 Losses': 'q2_losses',
        'Q3 Wins': 'q3_wins',
        'Q3 Losses': 'q3_losses',
        'Q4 Wins': 'q4_wins',
        'Q4 Losses': 'q4_losses',
        'Q1-Q2 Wins': 'q1_q2_wins',
        'Q1-Q2 Win Pct': 'q1_q2_win_pct',
        'Q3-Q4 Losses': 'q3_q4_losses',
        'Quality Score': 'quality_score',
        'Weighted Quality': 'weighted_quality',
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    season_cols = [c for c in df.columns if c not in ['team_code', 'team_name', 'conference']]
    print(f"  Prepared {len(season_cols)} season stat columns (including bart_rank)")
    
    return df


def aggregate_player_to_team(espn_player, espn_team):
    """Aggregate player-level stats to team-game level.
    
    Actual player_stats.csv columns:
    game_id, game_date, season, player_id, player_name, position, team, opponent,
    home_away, team_score, opponent_score, team_win, minutes_played, minutes_pct,
    points, field_goals_made, field_goals_attempted, three_pointers_made,
    three_pointers_attempted, free_throws_made, free_throws_attempted,
    offensive_rebounds, defensive_rebounds, assists, steals, blocks, turnovers, fouls
    """
    print("Aggregating player stats to team level...")
    
    df = espn_player.copy()
    
    # Handle minutes column - actual column is 'minutes_played'
    if 'minutes_played' in df.columns:
        df['minutes'] = df['minutes_played']
    elif 'minutes' not in df.columns:
        if 'min' in df.columns:
            df['minutes'] = df['min']
        else:
            df['minutes'] = 20
    
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
    if 'points' in df.columns:
        df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
    else:
        df['points'] = 0
    
    if 'pts_per_40' not in df.columns:
        df['pts_per_40'] = np.where(df['minutes'] > 0, df['points'] / df['minutes'] * 40, 0)
    
    agg_rows = []
    
    for (game_id, team), group in df.groupby(['game_id', 'team']):
        group = group.sort_values('minutes', ascending=False)
        
        row = {
            'game_id': game_id,
            'team': team,
        }
        
        if len(group) > 0:
            top_player = group.iloc[0]
            row['top_player_pts_per_40'] = top_player.get('pts_per_40', 0)
            row['top_player_usage'] = top_player.get('usage', 0) if 'usage' in top_player else 0
            row['top_player_ts'] = top_player.get('ts_pct', 0) if 'ts_pct' in top_player else 0
            row['top_player_minutes'] = top_player.get('minutes', 0)
        
        if len(group) >= 3:
            top3 = group.head(3)
            row['top3_avg_pts_per_40'] = top3['pts_per_40'].mean() if 'pts_per_40' in top3.columns else 0
            row['top3_avg_usage'] = top3['usage'].mean() if 'usage' in top3.columns else 0
        
        if len(group) >= 5:
            starters = group.head(5)
            row['starter_avg_pts_per_40'] = starters['pts_per_40'].mean() if 'pts_per_40' in starters.columns else 0
            if 'assists' in starters.columns and 'turnovers' in starters.columns:
                total_ast = starters['assists'].sum()
                total_tov = starters['turnovers'].sum()
                row['starter_avg_ast_to_tov'] = total_ast / max(total_tov, 1)
            if 'steals' in starters.columns and 'blocks' in starters.columns:
                row['starter_avg_stocks_per_40'] = (
                    (starters['steals'].sum() + starters['blocks'].sum()) / 
                    max(starters['minutes'].sum(), 1) * 40
                )
        
        if len(group) > 5:
            bench = group.iloc[5:]
            bench_pts = bench['points'].sum() if 'points' in bench.columns else 0
            bench_min = bench['minutes'].sum()
            row['bench_pts'] = bench_pts
            row['bench_min'] = bench_min
            row['bench_pts_per_min'] = bench_pts / max(bench_min, 1)
        
        if 'points' in group.columns and group['points'].sum() > 0:
            row['scoring_concentration'] = (group['points'] ** 2).sum() / (group['points'].sum() ** 2)
        if group['minutes'].sum() > 0:
            row['minutes_concentration'] = (group['minutes'] ** 2).sum() / (group['minutes'].sum() ** 2)
        
        agg_rows.append(row)
    
    agg_df = pd.DataFrame(agg_rows)
    print(f"  Aggregated {len(agg_df)} team-game rows from player data")
    
    return agg_df


def join_data(bart_games, bart_season, espn_team, player_agg):
    """Join all data sources into unified analysis dataframe."""
    print("Joining data sources...")
    
    # Create ESPN lookup by date+team for Vegas lines
    # ESPN team_stats has: game_date, team, spread, total_line
    espn_vegas = {}
    if espn_team is not None and len(espn_team) > 0:
        for _, row in espn_team.iterrows():
            date = str(row.get('game_date', ''))[:10]  # YYYY-MM-DD
            team = row.get('team', '')
            key = (date, team)
            espn_vegas[key] = {
                'spread': row.get('spread', np.nan),
                'total_line': row.get('total_line', np.nan),
            }
        print(f"  ESPN Vegas lookup: {len(espn_vegas)} entries")
    
    joined_rows = []
    
    for idx, game in bart_games.iterrows():
        for perspective in ['home', 'away']:
            if perspective == 'home':
                team_col = 'home_team'
                opp_col = 'away_team'
                score_col = 'home_score'
                opp_score_col = 'away_score'
                team_prefix = 'home_'
                opp_prefix = 'away_'
            else:
                team_col = 'away_team'
                opp_col = 'home_team'
                score_col = 'away_score'
                opp_score_col = 'home_score'
                team_prefix = 'away_'
                opp_prefix = 'home_'
            
            if team_col not in game or pd.isna(game[team_col]):
                continue
            
            row = {
                'game_id': game.get('id', game.get('game_id', idx)),
                'game_date': game.get('date', game.get('game_date', None)),
                'team': game[team_col],
                'opponent': game[opp_col],
                'is_home': 1 if perspective == 'home' else 0,
                'target_score': game.get(score_col, np.nan),
                'target_opponent_score': game.get(opp_score_col, np.nan),
            }
            
            # Get Vegas lines from ESPN data (match by date + team)
            game_date = str(game.get('date', ''))[:10]
            team_code = game[team_col]
            vegas_key = (game_date, team_code)
            
            if vegas_key in espn_vegas:
                vegas_data = espn_vegas[vegas_key]
                # spread in ESPN is from team's perspective already
                row['pregame_spread_line'] = vegas_data['spread']
                row['pregame_total_line'] = vegas_data['total_line']
            else:
                row['pregame_spread_line'] = np.nan
                row['pregame_total_line'] = np.nan
            
            rolling_cols = [c for c in bart_games.columns if '_rolling_' in c]
            for col in rolling_cols:
                if col.startswith(team_prefix):
                    new_name = 'pregame_team_' + col.replace(team_prefix, '')
                    row[new_name] = game[col]
                elif col.startswith(opp_prefix):
                    new_name = 'pregame_opp_' + col.replace(opp_prefix, '')
                    row[new_name] = game[col]
            
            context_mappings = [
                (f'{team_prefix}days_rest', 'pregame_team_days_rest'),
                (f'{opp_prefix}days_rest', 'pregame_opp_days_rest'),
                (f'{team_prefix}sos', 'pregame_team_sos'),
                (f'{opp_prefix}sos', 'pregame_opp_sos'),
                ('home_advantage', 'pregame_home_advantage'),
            ]
            for old_col, new_col in context_mappings:
                if old_col in game:
                    row[new_col] = game[old_col]
            
            team_code = game[team_col]  # e.g., "PENN", "NJIT"
            opp_code = game[opp_col]
            
            # Match on team_code (bart_games has codes like "MICH", bart_season has Team_Code -> team_code)
            team_season_match = bart_season[bart_season['team_code'] == team_code]
            if len(team_season_match) > 0:
                team_season_row = team_season_match.iloc[0]
                for col in bart_season.columns:
                    if col not in ['team_code', 'team_name', 'conference']:
                        row[f'pregame_team_season_{col}'] = team_season_row[col]
            
            opp_season_match = bart_season[bart_season['team_code'] == opp_code]
            if len(opp_season_match) > 0:
                opp_season_row = opp_season_match.iloc[0]
                for col in bart_season.columns:
                    if col not in ['team_code', 'team_name', 'conference']:
                        row[f'pregame_opp_season_{col}'] = opp_season_row[col]
            
            joined_rows.append(row)
    
    df = pd.DataFrame(joined_rows)
    
    if 'target_score' in df.columns and 'target_opponent_score' in df.columns:
        df['target_total'] = df['target_score'] + df['target_opponent_score']
        df['target_margin'] = df['target_score'] - df['target_opponent_score']
        df['target_win'] = (df['target_margin'] > 0).astype(int)
    
    if 'pregame_spread_line' in df.columns and 'target_margin' in df.columns:
        df['target_spread'] = df['target_margin'] + df['pregame_spread_line']
    
    print(f"  Joined dataset: {len(df)} rows")
    
    return df


def calculate_espn_rolling(df, espn_team, player_agg, window_sizes=[5, 10]):
    """Calculate rolling averages for ESPN stats."""
    print("Calculating ESPN rolling averages...")
    
    espn_cols_to_roll = [
        'field_goals_made', 'field_goals_attempted',
        'three_pointers_made', 'three_pointers_attempted', 
        'free_throws_made', 'free_throws_attempted',
        'offensive_rebounds', 'defensive_rebounds',
        'assists', 'steals', 'blocks', 'turnovers', 'fouls',
        'points_off_turnovers', 'fast_break_points', 'points_in_paint',
    ]
    
    espn_cols_to_roll = [c for c in espn_cols_to_roll if c in espn_team.columns]
    
    if 'efg_pct' not in espn_team.columns and 'field_goals_made' in espn_team.columns:
        espn_team['efg_pct'] = (
            (espn_team['field_goals_made'] + 0.5 * espn_team.get('three_pointers_made', 0)) /
            espn_team['field_goals_attempted'].replace(0, np.nan)
        )
    if 'efg_pct' in espn_team.columns:
        espn_cols_to_roll.append('efg_pct')
    
    if 'three_pt_pct' not in espn_team.columns and 'three_pointers_made' in espn_team.columns:
        espn_team['three_pt_pct'] = (
            espn_team['three_pointers_made'] / 
            espn_team['three_pointers_attempted'].replace(0, np.nan)
        )
    if 'three_pt_pct' in espn_team.columns:
        espn_cols_to_roll.append('three_pt_pct')
    
    espn_team = espn_team.sort_values(['team', 'game_date'] if 'game_date' in espn_team.columns else ['team', 'game_id'])
    
    rolling_dfs = []
    for team, group in espn_team.groupby('team'):
        group = group.copy()
        for col in espn_cols_to_roll:
            if col in group.columns:
                for window in window_sizes:
                    group[f'{col}_rolling_{window}'] = group[col].shift(1).rolling(window, min_periods=1).mean()
        rolling_dfs.append(group)
    
    espn_rolling = pd.concat(rolling_dfs, ignore_index=True)
    
    rolling_cols = [c for c in espn_rolling.columns if '_rolling_' in c]
    print(f"  Created {len(rolling_cols)} rolling stat columns")
    
    return espn_rolling, rolling_cols


def calculate_derived_features(df):
    """Calculate all derived features for modeling."""
    print("Calculating derived features...")
    
    # Expected tempo
    if 'pregame_team_season_tempo_raw' in df.columns and 'pregame_opp_season_tempo_raw' in df.columns:
        team_tempo_diff = df['pregame_team_season_tempo_raw'] - LEAGUE_AVG_TEMPO
        opp_tempo_diff = df['pregame_opp_season_tempo_raw'] - LEAGUE_AVG_TEMPO
        df['derived_expected_tempo'] = LEAGUE_AVG_TEMPO + team_tempo_diff + opp_tempo_diff
        df['derived_combined_tempo'] = df['derived_expected_tempo']
    
    # Expected offensive efficiency
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        team_off_diff = df['pregame_team_season_adjO'] - LEAGUE_AVG_EFFICIENCY
        opp_def_diff = df['pregame_opp_season_adjD'] - LEAGUE_AVG_EFFICIENCY
        df['derived_expected_off_eff'] = LEAGUE_AVG_EFFICIENCY + team_off_diff + opp_def_diff
    
    # Expected points
    if 'derived_expected_off_eff' in df.columns and 'derived_expected_tempo' in df.columns:
        df['derived_expected_points'] = df['derived_expected_off_eff'] * (df['derived_expected_tempo'] / 100)
    
    # Net efficiency
    if 'pregame_team_season_adjO' in df.columns and 'pregame_team_season_adjD' in df.columns:
        df['derived_team_net_eff'] = df['pregame_team_season_adjO'] - df['pregame_team_season_adjD']
    if 'pregame_opp_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_opp_net_eff'] = df['pregame_opp_season_adjO'] - df['pregame_opp_season_adjD']
    if 'derived_team_net_eff' in df.columns and 'derived_opp_net_eff' in df.columns:
        df['derived_net_eff_diff'] = df['derived_team_net_eff'] - df['derived_opp_net_eff']
    
    # Pythagorean
    if 'pregame_team_season_adjO' in df.columns and 'pregame_team_season_adjD' in df.columns:
        df['derived_team_pyth'] = df['pregame_team_season_adjO'] ** 11.5 / (
            df['pregame_team_season_adjO'] ** 11.5 + df['pregame_team_season_adjD'] ** 11.5
        )
    if 'pregame_opp_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_opp_pyth'] = df['pregame_opp_season_adjO'] ** 11.5 / (
            df['pregame_opp_season_adjO'] ** 11.5 + df['pregame_opp_season_adjD'] ** 11.5
        )
    if 'derived_team_pyth' in df.columns and 'derived_opp_pyth' in df.columns:
        df['derived_pyth_diff'] = df['derived_team_pyth'] - df['derived_opp_pyth']
    
    # Log5 probability
    if 'derived_team_pyth' in df.columns and 'derived_opp_pyth' in df.columns:
        p1 = df['derived_team_pyth']
        p2 = df['derived_opp_pyth']
        df['derived_log5_prob'] = (p1 * (1 - p2)) / (p1 * (1 - p2) + p2 * (1 - p1))
    
    # Additional derived features
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_adjO_adjD_product'] = df['pregame_team_season_adjO'] * df['pregame_opp_season_adjD'] / 100
    
    if 'derived_expected_tempo' in df.columns and 'pregame_team_season_adjO' in df.columns:
        df['derived_adjO_x_tempo'] = df['pregame_team_season_adjO'] * df['derived_expected_tempo'] / 100
    
    # Rank-based features (Barttorvik overall ranking)
    if 'pregame_team_season_bart_rank' in df.columns and 'pregame_opp_season_bart_rank' in df.columns:
        team_rank = df['pregame_team_season_bart_rank']
        opp_rank = df['pregame_opp_season_bart_rank']
        
        # Raw rank difference (positive = team ranked worse than opponent)
        df['derived_rank_diff'] = team_rank - opp_rank
        
        # Absolute rank difference
        df['derived_rank_diff_abs'] = df['derived_rank_diff'].abs()
        
        # Better and worse ranks
        df['derived_better_rank'] = df[['pregame_team_season_bart_rank', 'pregame_opp_season_bart_rank']].min(axis=1)
        df['derived_worse_rank'] = df[['pregame_team_season_bart_rank', 'pregame_opp_season_bart_rank']].max(axis=1)
        
        # Log rank ratio (higher = bigger mismatch)
        df['derived_log_rank_ratio'] = np.log(df['derived_worse_rank'] / df['derived_better_rank'].replace(0, 1))
        
        # Tier assignments (1=Elite, 2=Good, 3=Average, 4=Weak)
        def rank_to_tier(rank):
            if rank <= 25:
                return 1
            elif rank <= 75:
                return 2
            elif rank <= 150:
                return 3
            else:
                return 4
        
        df['derived_team_tier'] = team_rank.apply(rank_to_tier)
        df['derived_opp_tier'] = opp_rank.apply(rank_to_tier)
        df['derived_tier_diff'] = df['derived_opp_tier'] - df['derived_team_tier']  # positive = team in better tier
        
        # Binary mismatch indicators
        df['derived_is_mismatch_50'] = (df['derived_rank_diff_abs'] >= 50).astype(int)
        df['derived_is_mismatch_100'] = (df['derived_rank_diff_abs'] >= 100).astype(int)
        
        # Elite vs weaker teams
        better_rank = df['derived_better_rank']
        worse_rank = df['derived_worse_rank']
        df['derived_is_elite_vs_avg'] = ((better_rank <= 25) & (worse_rank > 75)).astype(int)
        df['derived_is_top50_vs_weak'] = ((better_rank <= 50) & (worse_rank > 150)).astype(int)
        
        print(f"  Created rank-based derived features")
    
    # Quad-based features
    if 'pregame_team_season_weighted_quality' in df.columns and 'pregame_opp_season_weighted_quality' in df.columns:
        # Weighted quality differential (strongest quad predictor, r=0.741)
        df['derived_weighted_quality_diff'] = (
            df['pregame_team_season_weighted_quality'] - df['pregame_opp_season_weighted_quality']
        )
        
        # Quality score differential
        if 'pregame_team_season_quality_score' in df.columns and 'pregame_opp_season_quality_score' in df.columns:
            df['derived_quality_score_diff'] = (
                df['pregame_team_season_quality_score'] - df['pregame_opp_season_quality_score']
            )
        
        # Q1-Q2 win percentage differential
        if 'pregame_team_season_q1_q2_win_pct' in df.columns and 'pregame_opp_season_q1_q2_win_pct' in df.columns:
            df['derived_q1_q2_win_pct_diff'] = (
                df['pregame_team_season_q1_q2_win_pct'] - df['pregame_opp_season_q1_q2_win_pct']
            )
        
        # Q3-Q4 losses differential (negative = team has fewer bad losses)
        if 'pregame_team_season_q3_q4_losses' in df.columns and 'pregame_opp_season_q3_q4_losses' in df.columns:
            df['derived_q3_q4_losses_diff'] = (
                df['pregame_team_season_q3_q4_losses'] - df['pregame_opp_season_q3_q4_losses']
            )
        
        # Q1 wins differential
        if 'pregame_team_season_q1_wins' in df.columns and 'pregame_opp_season_q1_wins' in df.columns:
            df['derived_q1_wins_diff'] = (
                df['pregame_team_season_q1_wins'] - df['pregame_opp_season_q1_wins']
            )
        
        print(f"  Created quad-based derived features")
    
    derived_cols = [c for c in df.columns if c.startswith('derived_')]
    print(f"  Created {len(derived_cols)} derived features")
    
    return df


def build_full_dataset(bart_games_path, bart_season_path, espn_team_path, espn_player_path):
    """Build the complete dataset with all features."""
    print("=" * 80)
    print("BUILDING FULL DATASET")
    print("=" * 80)
    
    # Load data
    bart_games, bart_season, espn_team, espn_player = load_data(
        bart_games_path, bart_season_path, espn_team_path, espn_player_path
    )
    
    # Prepare season data
    bart_season = prepare_season_data(bart_season)
    
    # Aggregate player stats
    player_agg = aggregate_player_to_team(espn_player, espn_team)
    
    # Join data
    df = join_data(bart_games, bart_season, espn_team, player_agg)
    
    # Calculate derived features
    df = calculate_derived_features(df)
    
    print(f"\n  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Show column counts by type
    pregame_team = [c for c in df.columns if c.startswith('pregame_team_')]
    pregame_opp = [c for c in df.columns if c.startswith('pregame_opp_')]
    derived = [c for c in df.columns if c.startswith('derived_')]
    targets = [c for c in df.columns if c.startswith('target_')]
    
    print(f"  Pregame team features: {len(pregame_team)}")
    print(f"  Pregame opponent features: {len(pregame_opp)}")
    print(f"  Derived features: {len(derived)}")
    print(f"  Target columns: {len(targets)}")
    
    return df


def select_features():
    """
    Select non-redundant features based on Phase 0 analysis.
    """
    
    # Core derived features (best performers)
    derived_core = [
        'derived_expected_points',
        'derived_net_eff_diff',
        'derived_expected_tempo',
        'derived_log5_prob',
        'derived_pyth_diff',
    ]
    
    # Vegas lines
    vegas_features = [
        'pregame_spread_line',
        'pregame_total_line',
    ]
    
    # Team season stats
    team_season = [
        'pregame_team_season_adjO',
        'pregame_team_season_adjD',
        'pregame_team_season_tempo_raw',
        'pregame_team_season_efg_off',
        'pregame_team_season_tov_off',
        'pregame_team_season_oreb_pct',
        'pregame_team_season_ftr_off',
        'pregame_team_season_3pt_off',
    ]
    
    # Opponent season stats
    opp_season = [
        'pregame_opp_season_adjO',
        'pregame_opp_season_adjD',
        'pregame_opp_season_tempo_raw',
        'pregame_opp_season_efg_def',
        'pregame_opp_season_tov_def',
        'pregame_opp_season_dreb_pct',
        'pregame_opp_season_3pt_def',
    ]
    
    # Rolling stats
    rolling_stats = [
        'pregame_team_AdjO_rolling_10',
        'pregame_team_AdjD_rolling_10',
        'pregame_opp_AdjO_rolling_10',
        'pregame_opp_AdjD_rolling_10',
    ]
    
    # Context features
    context = [
        'is_home',
        'pregame_team_days_rest',
        'pregame_opp_days_rest',
        'pregame_team_sos',
    ]
    
    # Rank-based features (Barttorvik overall ranking)
    rank_features = [
        'derived_rank_diff',           # Raw rank difference (r=-0.573 with margin)
        'derived_tier_diff',           # Tier difference (1-4 scale)
        'derived_log_rank_ratio',      # Log of rank ratio for mismatches
        'derived_is_mismatch_100',     # Binary: 100+ rank difference
    ]
    
    # Quad-based features (strongest predictors)
    quad_features = [
        'derived_weighted_quality_diff',  # Best single predictor (r=0.741 with margin)
        'derived_q1_q2_win_pct_diff',     # Quality win percentage diff (r=0.579)
        'derived_q3_q4_losses_diff',      # Bad losses diff (r=-0.659)
    ]
    
    all_features = (
        derived_core + 
        vegas_features + 
        team_season + 
        opp_season + 
        rolling_stats + 
        context +
        rank_features +
        quad_features
    )
    
    return all_features


def select_features_no_vegas():
    """Select features for modeling WITHOUT Vegas lines.
    
    Use this for:
    - Early predictions before lines are released
    - Training on full historical data (which lacks Vegas)
    - Getting a "pure" model view uninfluenced by Vegas
    """
    
    # Core derived features (best performers)
    derived_core = [
        'derived_expected_points',
        'derived_net_eff_diff',
        'derived_expected_tempo',
        'derived_log5_prob',
        'derived_pyth_diff',
    ]
    
    # Team season stats
    team_season = [
        'pregame_team_season_adjO',
        'pregame_team_season_adjD',
        'pregame_team_season_tempo_raw',
        'pregame_team_season_efg_off',
        'pregame_team_season_tov_off',
        'pregame_team_season_oreb_pct',
        'pregame_team_season_ftr_off',
        'pregame_team_season_3pt_off',
    ]
    
    # Opponent season stats
    opp_season = [
        'pregame_opp_season_adjO',
        'pregame_opp_season_adjD',
        'pregame_opp_season_tempo_raw',
        'pregame_opp_season_efg_def',
        'pregame_opp_season_tov_def',
        'pregame_opp_season_dreb_pct',
        'pregame_opp_season_3pt_def',
    ]
    
    # Rolling stats
    rolling_stats = [
        'pregame_team_AdjO_rolling_10',
        'pregame_team_AdjD_rolling_10',
        'pregame_opp_AdjO_rolling_10',
        'pregame_opp_AdjD_rolling_10',
    ]
    
    # Context features
    context = [
        'is_home',
        'pregame_team_days_rest',
        'pregame_opp_days_rest',
        'pregame_team_sos',
    ]
    
    # Rank-based features (Barttorvik overall ranking)
    rank_features = [
        'derived_rank_diff',           # Raw rank difference (r=-0.573 with margin)
        'derived_tier_diff',           # Tier difference (1-4 scale)
        'derived_log_rank_ratio',      # Log of rank ratio for mismatches
        'derived_is_mismatch_100',     # Binary: 100+ rank difference
    ]
    
    # Quad-based features (strongest predictors)
    quad_features = [
        'derived_weighted_quality_diff',  # Best single predictor (r=0.741 with margin)
        'derived_q1_q2_win_pct_diff',     # Quality win percentage diff (r=0.579)
        'derived_q3_q4_losses_diff',      # Bad losses diff (r=-0.659)
    ]
    
    all_features = (
        derived_core + 
        team_season + 
        opp_season + 
        rolling_stats + 
        context +
        rank_features +
        quad_features
    )
    
    return all_features


def check_feature_availability(df, features):
    """Check which features are available in the dataframe."""
    available = []
    missing = []
    
    for f in features:
        if f in df.columns:
            available.append(f)
        else:
            missing.append(f)
    
    return available, missing


def train_model(X_train, y_train, X_test, y_test, model_type='xgb'):
    """Train a model and return predictions."""
    
    if model_type == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return model, train_pred, test_pred


def evaluate_regression(y_true, y_pred, name="Model"):
    """Evaluate regression model performance."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Correlation
    r, _ = pearsonr(y_true, y_pred)
    
    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'correlation': r
    }


def evaluate_spread_prediction(y_true_margin, y_pred_margin, spread_line):
    """Evaluate spread prediction performance."""
    # Predicted cover: margin > -spread (or margin + spread > 0)
    pred_cover = (y_pred_margin + spread_line) > 0
    actual_cover = (y_true_margin + spread_line) > 0
    
    accuracy = accuracy_score(actual_cover, pred_cover)
    
    return accuracy


def print_feature_importance(model, feature_names, top_n=20):
    """Print feature importance from the model."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print(f"\n  Top {top_n} Feature Importances:")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"    {i+1:2d}. {feature_names[idx]:50s} {importance[idx]:.4f}")


def run_model_comparison(df, features, target_col, target_name, vegas_col=None):
    """Run full model comparison for a single target."""
    
    print(f"\n{'='*80}")
    print(f"MODEL: {target_name.upper()}")
    print(f"{'='*80}")
    
    # Check feature availability
    available_features, missing_features = check_feature_availability(df, features)
    
    if missing_features:
        print(f"\n  Missing features ({len(missing_features)}):")
        for f in missing_features[:10]:
            print(f"    - {f}")
        if len(missing_features) > 10:
            print(f"    ... and {len(missing_features) - 10} more")
    
    print(f"\n  Using {len(available_features)} features")
    
    # Prepare data
    model_df = df[[target_col] + available_features].dropna()
    print(f"  Samples after dropping NA: {len(model_df)}")
    
    if len(model_df) < 100:
        print("  ERROR: Not enough samples!")
        return None
    
    X = model_df[available_features]
    y = model_df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    train_metrics = evaluate_regression(y_train, train_pred, "Train")
    test_metrics = evaluate_regression(y_test, test_pred, "Test")
    
    print(f"\n  Results:")
    print(f"    {'Metric':<15} {'Train':>10} {'Test':>10}")
    print(f"    {'-'*35}")
    print(f"    {'MAE':<15} {train_metrics['mae']:>10.2f} {test_metrics['mae']:>10.2f}")
    print(f"    {'RMSE':<15} {train_metrics['rmse']:>10.2f} {test_metrics['rmse']:>10.2f}")
    print(f"    {'Correlation':<15} {train_metrics['correlation']:>10.3f} {test_metrics['correlation']:>10.3f}")
    
    # Compare to Vegas if available
    if vegas_col and vegas_col in df.columns:
        # Get Vegas predictions for test set
        test_indices = X_test.index
        vegas_pred = df.loc[test_indices, vegas_col]
        
        # For spread, Vegas predicts margin as -spread
        if 'spread' in vegas_col.lower():
            vegas_pred_margin = -vegas_pred
            vegas_metrics = evaluate_regression(y_test, vegas_pred_margin, "Vegas")
        else:
            # For total, we need to compare to actual total
            # But Vegas total is for both teams combined
            # Our target might be single team score
            if 'total' in target_name.lower():
                # Vegas total / 2 as rough estimate for single team
                vegas_pred_score = vegas_pred / 2
                vegas_metrics = evaluate_regression(y_test, vegas_pred_score, "Vegas/2")
            else:
                vegas_metrics = evaluate_regression(y_test, vegas_pred, "Vegas")
        
        print(f"\n  Vegas Comparison:")
        print(f"    Vegas MAE:  {vegas_metrics['mae']:.2f}")
        print(f"    Model MAE:  {test_metrics['mae']:.2f}")
        print(f"    Improvement: {vegas_metrics['mae'] - test_metrics['mae']:.2f} points")
    
    # Feature importance
    print_feature_importance(model, available_features)
    
    return {
        'model': model,
        'features': available_features,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'test_predictions': test_pred,
        'test_actual': y_test
    }


def run_spread_betting_analysis(df, model_result, features):
    """Analyze spread betting performance."""
    
    print(f"\n{'='*80}")
    print("SPREAD BETTING ANALYSIS")
    print(f"{'='*80}")
    
    if model_result is None:
        print("  No model result available")
        return
    
    # Get test set
    test_indices = model_result['test_actual'].index
    
    if 'pregame_spread_line' not in df.columns:
        print("  No spread line available")
        return
    
    spread_line = df.loc[test_indices, 'pregame_spread_line']
    actual_margin = model_result['test_actual']
    pred_margin = model_result['test_predictions']
    
    # Calculate covers
    actual_cover = (actual_margin + spread_line) > 0
    
    # Model predicted cover (using predicted margin)
    model_pred_cover = (pred_margin + spread_line) > 0
    
    # Accuracy
    model_accuracy = accuracy_score(actual_cover, model_pred_cover)
    
    # Baseline: always pick favorite (spread < 0 means favorite)
    favorite_cover = spread_line < 0
    favorite_accuracy = accuracy_score(actual_cover, favorite_cover)
    
    print(f"\n  Cover Prediction Accuracy:")
    print(f"    Model:          {100*model_accuracy:.1f}%")
    print(f"    Always Favorite: {100*favorite_accuracy:.1f}%")
    print(f"    Random (50%):   50.0%")
    
    # Confidence-based betting
    print(f"\n  Confidence Analysis:")
    
    # Model confidence = |predicted_margin - (-spread)|
    # Higher = more confident the model disagrees with Vegas
    model_edge = pred_margin - (-spread_line)  # Positive = model thinks team covers
    
    # Bin by confidence
    confidence_bins = [
        (0, 2, "Low (0-2 pts)"),
        (2, 5, "Medium (2-5 pts)"),
        (5, 10, "High (5-10 pts)"),
        (10, 100, "Very High (10+ pts)")
    ]
    
    for low, high, label in confidence_bins:
        mask = (abs(model_edge) >= low) & (abs(model_edge) < high)
        if mask.sum() > 10:
            bin_accuracy = accuracy_score(
                actual_cover[mask], 
                model_pred_cover[mask]
            )
            print(f"    {label:25s}: {100*bin_accuracy:.1f}% ({mask.sum()} games)")


def run_total_betting_analysis(df, features):
    """Analyze over/under betting performance."""
    
    print(f"\n{'='*80}")
    print("TOTAL (OVER/UNDER) BETTING ANALYSIS")
    print(f"{'='*80}")
    
    # Check required columns
    required = ['target_total', 'pregame_total_line']
    if not all(c in df.columns for c in required):
        print("  Missing required columns")
        return None
    
    # Get available features
    available_features, _ = check_feature_availability(df, features)
    
    # Remove pregame_total_line from features since we select it explicitly as target comparison
    features_without_total = [f for f in available_features if f != 'pregame_total_line']
    
    # Prepare data - reset index to avoid duplicate index issues
    model_df = df[['target_total', 'pregame_total_line'] + features_without_total].dropna().reset_index(drop=True)
    print(f"  Samples: {len(model_df)}")
    
    X = model_df[features_without_total]
    y = model_df['target_total']
    vegas_total = model_df['pregame_total_line']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vegas_test = vegas_total.loc[X_test.index]
    
    # Train model
    model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    model_mae = mean_absolute_error(y_test, test_pred)
    vegas_mae = mean_absolute_error(y_test, vegas_test)
    
    print(f"\n  MAE Comparison:")
    print(f"    Model MAE: {model_mae:.2f}")
    print(f"    Vegas MAE: {vegas_mae:.2f}")
    print(f"    Improvement: {vegas_mae - model_mae:.2f} points")
    
    # Over/Under accuracy
    actual_over = y_test > vegas_test
    model_pred_over = test_pred > vegas_test.values
    
    model_ou_accuracy = accuracy_score(actual_over, model_pred_over)
    
    print(f"\n  Over/Under Prediction:")
    print(f"    Model accuracy: {100*model_ou_accuracy:.1f}%")
    print(f"    Random (50%):   50.0%")
    
    # Feature importance
    print_feature_importance(model, available_features)
    
    return {
        'model': model,
        'mae': model_mae,
        'vegas_mae': vegas_mae,
        'ou_accuracy': model_ou_accuracy
    }


def run_moneyline_analysis(df, margin_model_result):
    """Analyze straight-up win prediction performance."""
    
    print(f"\n{'='*80}")
    print("MONEYLINE (WIN/LOSS) ANALYSIS")
    print(f"{'='*80}")
    
    if margin_model_result is None:
        print("  No margin model result available")
        return None
    
    # Get test set predictions
    test_indices = margin_model_result['test_actual'].index
    actual_margin = margin_model_result['test_actual']
    pred_margin = margin_model_result['test_predictions']
    
    # Actual wins (margin > 0)
    actual_win = (actual_margin > 0).astype(int)
    
    # Predicted wins (predicted margin > 0)
    pred_win = (pred_margin > 0).astype(int)
    
    # Accuracy
    win_accuracy = accuracy_score(actual_win, pred_win)
    
    # Favorite baseline - if we have spread, favorite is team with negative spread
    if 'pregame_spread_line' in df.columns:
        spread_line = df.loc[test_indices, 'pregame_spread_line']
        # Favorite = spread < 0 (they're expected to win by X points)
        favorite_pred = (spread_line < 0).astype(int)
        favorite_accuracy = accuracy_score(actual_win, favorite_pred)
    else:
        favorite_accuracy = 0.5
    
    print(f"\n  Win Prediction Accuracy:")
    print(f"    Model:          {100*win_accuracy:.1f}%")
    print(f"    Pick Favorite:  {100*favorite_accuracy:.1f}%")
    print(f"    Random (50%):   50.0%")
    
    # Confidence analysis - higher |predicted margin| = more confident
    print(f"\n  Confidence Analysis (by predicted margin):")
    
    confidence_bins = [
        (0, 3, "Low (0-3 pts)"),
        (3, 7, "Medium (3-7 pts)"),
        (7, 12, "High (7-12 pts)"),
        (12, 100, "Very High (12+ pts)")
    ]
    
    for low, high, label in confidence_bins:
        mask = (abs(pred_margin) >= low) & (abs(pred_margin) < high)
        if mask.sum() > 10:
            bin_accuracy = accuracy_score(actual_win[mask], pred_win[mask])
            print(f"    {label:25s}: {100*bin_accuracy:.1f}% ({mask.sum()} games)")
    
    # Upset detection - when model predicts opposite of favorite
    if 'pregame_spread_line' in df.columns:
        upset_pred = pred_win != favorite_pred
        upset_mask = upset_pred & (abs(pred_margin) > 3)  # Confident upsets only
        if upset_mask.sum() > 0:
            upset_accuracy = accuracy_score(actual_win[upset_mask], pred_win[upset_mask])
            print(f"\n  Upset Predictions (model disagrees with favorite, margin > 3):")
            print(f"    Accuracy: {100*upset_accuracy:.1f}% ({upset_mask.sum()} games)")
    
    return {
        'win_accuracy': win_accuracy,
        'favorite_accuracy': favorite_accuracy
    }


def diagnose_vegas_data(df, bart_games, espn_team):
    """Diagnose why Vegas data might not be matching."""
    
    print(f"\n{'='*80}")
    print("VEGAS DATA DIAGNOSTIC")
    print(f"{'='*80}")
    
    # Check ESPN data
    print(f"\n  ESPN team_stats:")
    print(f"    Total rows: {len(espn_team)}")
    print(f"    Has 'spread' column: {'spread' in espn_team.columns}")
    print(f"    Has 'total_line' column: {'total_line' in espn_team.columns}")
    
    if 'spread' in espn_team.columns:
        non_null_spread = espn_team['spread'].notna().sum()
        print(f"    Non-null spreads: {non_null_spread}")
    
    if 'game_date' in espn_team.columns:
        print(f"    Date range: {espn_team['game_date'].min()} to {espn_team['game_date'].max()}")
    
    # Check bart_games data
    print(f"\n  Barttorvik games:")
    print(f"    Total rows: {len(bart_games)}")
    if 'date' in bart_games.columns:
        print(f"    Date range: {bart_games['date'].min()} to {bart_games['date'].max()}")
    
    # Check joined data
    print(f"\n  Joined dataset:")
    print(f"    Total rows: {len(df)}")
    
    if 'pregame_spread_line' in df.columns:
        has_spread = df['pregame_spread_line'].notna().sum()
        print(f"    Rows with spread: {has_spread} ({100*has_spread/len(df):.1f}%)")
    
    if 'pregame_total_line' in df.columns:
        has_total = df['pregame_total_line'].notna().sum()
        print(f"    Rows with total_line: {has_total} ({100*has_total/len(df):.1f}%)")
    
    # Sample of unmatched games
    if 'pregame_spread_line' in df.columns:
        unmatched = df[df['pregame_spread_line'].isna()].head(10)
        if len(unmatched) > 0:
            print(f"\n  Sample unmatched games (no Vegas data):")
            for _, row in unmatched.iterrows():
                print(f"    {row.get('game_date', 'N/A')} - {row.get('team', 'N/A')} vs {row.get('opponent', 'N/A')}")
    
    # Check team code matching
    bart_teams = set(bart_games['home_team'].unique()) | set(bart_games['away_team'].unique())
    espn_teams = set(espn_team['team'].unique()) if 'team' in espn_team.columns else set()
    
    common_teams = bart_teams & espn_teams
    bart_only = bart_teams - espn_teams
    espn_only = espn_teams - bart_teams
    
    print(f"\n  Team code matching:")
    print(f"    Barttorvik teams: {len(bart_teams)}")
    print(f"    ESPN teams: {len(espn_teams)}")
    print(f"    Common: {len(common_teams)}")
    print(f"    Bart only: {len(bart_only)}")
    print(f"    ESPN only: {len(espn_only)}")
    
    if bart_only and len(bart_only) <= 20:
        print(f"    Bart-only teams: {sorted(bart_only)[:20]}")
    if espn_only and len(espn_only) <= 20:
        print(f"    ESPN-only teams: {sorted(espn_only)[:20]}")


def run_holdout_season_validation(df, features, holdout_season='2025_26'):
    """Run validation using a held-out season."""
    
    print(f"\n{'='*80}")
    print(f"HELD-OUT SEASON VALIDATION (Test on {holdout_season})")
    print(f"{'='*80}")
    
    # Check for season column
    if 'game_date' not in df.columns:
        print("  No game_date column found")
        return None
    
    # Parse season from date
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    
    # Season logic: games before July are previous season, after July are current season
    # e.g., 2024-11 to 2025-04 is 2024_25 season
    df['year'] = df['game_date'].dt.year
    df['month'] = df['game_date'].dt.month
    df['season'] = df.apply(
        lambda r: f"{r['year']-1}_{str(r['year'])[-2:]}" if r['month'] < 7 else f"{r['year']}_{str(r['year']+1)[-2:]}", 
        axis=1
    )
    
    seasons = df['season'].value_counts().sort_index()
    print(f"\n  Seasons in data:")
    for season, count in seasons.items():
        print(f"    {season}: {count} rows")
    
    # Split by season
    train_df = df[df['season'] != holdout_season]
    test_df = df[df['season'] == holdout_season]
    
    print(f"\n  Train seasons: {sorted(train_df['season'].unique().tolist())}")
    print(f"  Test season: {holdout_season}")
    print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    
    if len(test_df) < 100:
        print(f"  WARNING: Only {len(test_df)} test samples, results may be unreliable")
        if len(test_df) == 0:
            return None
    
    # Get available features
    available_features, _ = check_feature_availability(df, features)
    
    # --- MARGIN MODEL ---
    print(f"\n  MARGIN (SPREAD) PREDICTION:")
    
    cols_needed = ['target_margin'] + available_features
    train_margin = train_df[cols_needed].dropna()
    test_margin = test_df[cols_needed].dropna()
    
    print(f"    Train samples: {len(train_margin)}, Test samples: {len(test_margin)}")
    
    if len(test_margin) > 50:
        X_train = train_margin[available_features]
        y_train = train_margin['target_margin']
        X_test = test_margin[available_features]
        y_test = test_margin['target_margin']
        
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        margin_mae = mean_absolute_error(y_test, test_pred)
        margin_corr = np.corrcoef(y_test, test_pred)[0, 1]
        
        print(f"    Test MAE: {margin_mae:.2f}")
        print(f"    Test Correlation: {margin_corr:.3f}")
        
        # Win accuracy
        actual_win = (y_test > 0).astype(int)
        pred_win = (test_pred > 0).astype(int)
        win_acc = accuracy_score(actual_win, pred_win)
        print(f"    Win Prediction Accuracy: {100*win_acc:.1f}%")
        
        # Spread cover if available
        if 'pregame_spread_line' in test_df.columns:
            spread_test = test_df.loc[test_margin.index, 'pregame_spread_line']
            if spread_test.notna().sum() > 50:
                valid_spread = spread_test.notna()
                actual_cover = (y_test[valid_spread] + spread_test[valid_spread]) > 0
                pred_cover = (test_pred[valid_spread.values] + spread_test[valid_spread].values) > 0
                cover_acc = accuracy_score(actual_cover, pred_cover)
                print(f"    Spread Cover Accuracy: {100*cover_acc:.1f}% ({valid_spread.sum()} games with lines)")
    
    # --- TOTAL MODEL ---
    print(f"\n  TOTAL (OVER/UNDER) PREDICTION:")
    
    # Remove total_line from features for total prediction
    total_features = [f for f in available_features if f != 'pregame_total_line']
    cols_needed = ['target_total', 'pregame_total_line'] + total_features
    cols_available = [c for c in cols_needed if c in df.columns]
    
    train_total = train_df[cols_available].dropna()
    test_total = test_df[cols_available].dropna()
    
    print(f"    Train samples: {len(train_total)}, Test samples: {len(test_total)}")
    
    if len(test_total) > 50 and 'pregame_total_line' in cols_available:
        X_train = train_total[total_features]
        y_train = train_total['target_total']
        X_test = test_total[total_features]
        y_test = test_total['target_total']
        vegas_test = test_total['pregame_total_line']
        
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        total_mae = mean_absolute_error(y_test, test_pred)
        vegas_mae = mean_absolute_error(y_test, vegas_test)
        
        print(f"    Model MAE: {total_mae:.2f}")
        print(f"    Vegas MAE: {vegas_mae:.2f}")
        
        # Over/under accuracy
        actual_over = y_test > vegas_test
        pred_over = test_pred > vegas_test.values
        ou_acc = accuracy_score(actual_over, pred_over)
        print(f"    Over/Under Accuracy: {100*ou_acc:.1f}%")
    
    return True


def run_temporal_validation(df, features, test_days=6):
    """Run time-based validation using only games with Vegas data.
    
    Splits by date: trains on earlier games, tests on most recent games.
    This is the proper way to validate when we only have Vegas data for a short period.
    """
    
    print(f"\n{'='*80}")
    print(f"TEMPORAL VALIDATION (Last {test_days} days as test set)")
    print(f"{'='*80}")
    
    # Filter to games with Vegas data
    if 'pregame_spread_line' not in df.columns:
        print("  No spread data available")
        return None
    
    vegas_df = df[df['pregame_spread_line'].notna()].copy()
    print(f"\n  Games with Vegas data: {len(vegas_df)} rows")
    
    if len(vegas_df) < 200:
        print("  Not enough Vegas data for temporal validation")
        return None
    
    # Parse dates
    vegas_df['game_date'] = pd.to_datetime(vegas_df['game_date'], errors='coerce')
    vegas_df = vegas_df.dropna(subset=['game_date'])
    
    # Get date range
    min_date = vegas_df['game_date'].min()
    max_date = vegas_df['game_date'].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)
    
    print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"  Train: before {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Test: {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Split
    train_df = vegas_df[vegas_df['game_date'] < cutoff_date]
    test_df = vegas_df[vegas_df['game_date'] >= cutoff_date]
    
    print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    
    if len(test_df) < 50:
        print("  WARNING: Small test set, results may be unreliable")
    
    # Get available features
    available_features, _ = check_feature_availability(df, features)
    
    results = {}
    
    # --- MARGIN MODEL ---
    print(f"\n  MARGIN (SPREAD) PREDICTION:")
    
    # Remove spread from features to avoid duplicate column selection
    margin_features = [f for f in available_features if f != 'pregame_spread_line']
    cols_needed = ['target_margin', 'pregame_spread_line'] + margin_features
    cols_available = [c for c in cols_needed if c in vegas_df.columns]
    
    train_margin = train_df[cols_available].dropna().reset_index(drop=True)
    test_margin = test_df[cols_available].dropna().reset_index(drop=True)
    
    print(f"    Train samples: {len(train_margin)}, Test samples: {len(test_margin)}")
    
    if len(test_margin) > 30:
        X_train = train_margin[margin_features]
        y_train = train_margin['target_margin']
        X_test = test_margin[margin_features]
        y_test = test_margin['target_margin']
        spread_test = test_margin['pregame_spread_line']
        
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        # Metrics
        margin_mae = mean_absolute_error(y_test, test_pred)
        vegas_mae = mean_absolute_error(y_test, -spread_test)  # Vegas prediction is -spread
        margin_corr = np.corrcoef(y_test, test_pred)[0, 1]
        
        print(f"    Model MAE: {margin_mae:.2f}")
        print(f"    Vegas MAE: {vegas_mae:.2f}")
        print(f"    Correlation: {margin_corr:.3f}")
        
        # Win prediction
        actual_win = (y_test > 0).astype(int)
        pred_win = (test_pred > 0).astype(int)
        win_acc = accuracy_score(actual_win, pred_win)
        print(f"    Win Accuracy: {100*win_acc:.1f}%")
        
        # Spread cover
        actual_cover = (y_test + spread_test) > 0
        pred_cover = (test_pred + spread_test.values) > 0
        cover_acc = accuracy_score(actual_cover, pred_cover)
        print(f"    Spread Cover Accuracy: {100*cover_acc:.1f}%")
        
        results['margin'] = {
            'mae': margin_mae,
            'vegas_mae': vegas_mae,
            'win_accuracy': win_acc,
            'cover_accuracy': cover_acc,
            'model': model
        }
    
    # --- TOTAL MODEL ---
    print(f"\n  TOTAL (OVER/UNDER) PREDICTION:")
    
    total_features = [f for f in available_features if f != 'pregame_total_line']
    cols_needed = ['target_total', 'pregame_total_line'] + total_features
    cols_available = [c for c in cols_needed if c in vegas_df.columns]
    
    train_total = train_df[cols_available].dropna().reset_index(drop=True)
    test_total = test_df[cols_available].dropna().reset_index(drop=True)
    
    print(f"    Train samples: {len(train_total)}, Test samples: {len(test_total)}")
    
    if len(test_total) > 30:
        X_train = train_total[total_features]
        y_train = train_total['target_total']
        X_test = test_total[total_features]
        y_test = test_total['target_total']
        vegas_test = test_total['pregame_total_line']
        
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        # Metrics
        total_mae = mean_absolute_error(y_test, test_pred)
        vegas_mae = mean_absolute_error(y_test, vegas_test)
        
        print(f"    Model MAE: {total_mae:.2f}")
        print(f"    Vegas MAE: {vegas_mae:.2f}")
        
        # Over/under accuracy
        actual_over = y_test > vegas_test
        pred_over = test_pred > vegas_test.values
        ou_acc = accuracy_score(actual_over, pred_over)
        print(f"    Over/Under Accuracy: {100*ou_acc:.1f}%")
        
        results['total'] = {
            'mae': total_mae,
            'vegas_mae': vegas_mae,
            'ou_accuracy': ou_acc,
            'model': model
        }
    
    return results


def train_no_vegas_models(df, features_no_vegas):
    """Train models without Vegas features on ALL available data."""
    
    print(f"\n{'='*80}")
    print("TRAINING NO-VEGAS MODELS (Full Dataset)")
    print(f"{'='*80}")
    
    # Get available features
    available_features, missing = check_feature_availability(df, features_no_vegas)
    print(f"\n  Features: {len(available_features)} available, {len(missing)} missing")
    
    results = {}
    
    # --- SCORE MODEL ---
    print(f"\n  SCORE MODEL:")
    cols_needed = ['target_score'] + available_features
    model_df = df[cols_needed].dropna().reset_index(drop=True)
    print(f"    Samples: {len(model_df)}")
    
    if len(model_df) > 100:
        X = model_df[available_features]
        y = model_df['target_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        mae = mean_absolute_error(y_test, test_pred)
        corr = np.corrcoef(y_test, test_pred)[0, 1]
        print(f"    Test MAE: {mae:.2f}, Correlation: {corr:.3f}")
        
        results['score'] = {'model': model, 'mae': mae, 'correlation': corr}
    
    # --- MARGIN MODEL ---
    print(f"\n  MARGIN MODEL:")
    cols_needed = ['target_margin'] + available_features
    model_df = df[cols_needed].dropna().reset_index(drop=True)
    print(f"    Samples: {len(model_df)}")
    
    if len(model_df) > 100:
        X = model_df[available_features]
        y = model_df['target_margin']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        mae = mean_absolute_error(y_test, test_pred)
        corr = np.corrcoef(y_test, test_pred)[0, 1]
        
        # Win accuracy
        actual_win = (y_test > 0).astype(int)
        pred_win = (test_pred > 0).astype(int)
        win_acc = accuracy_score(actual_win, pred_win)
        
        print(f"    Test MAE: {mae:.2f}, Correlation: {corr:.3f}")
        print(f"    Win Accuracy: {100*win_acc:.1f}%")
        
        results['margin'] = {'model': model, 'mae': mae, 'correlation': corr, 'win_accuracy': win_acc}
    
    # --- TOTAL MODEL ---
    print(f"\n  TOTAL MODEL:")
    cols_needed = ['target_total'] + available_features
    model_df = df[cols_needed].dropna().reset_index(drop=True)
    print(f"    Samples: {len(model_df)}")
    
    if len(model_df) > 100:
        X = model_df[available_features]
        y = model_df['target_total']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, train_pred, test_pred = train_model(X_train, y_train, X_test, y_test)
        
        mae = mean_absolute_error(y_test, test_pred)
        corr = np.corrcoef(y_test, test_pred)[0, 1]
        print(f"    Test MAE: {mae:.2f}, Correlation: {corr:.3f}")
        
        results['total'] = {'model': model, 'mae': mae, 'correlation': corr}
    
    return results, available_features


def save_models(results, feature_list, output_dir, suffix=''):
    """Save trained models to JSON format for later use.
    
    Args:
        results: dict of model results
        feature_list: list of features used
        output_dir: directory to save to
        suffix: optional suffix for filenames (e.g., '_no_vegas')
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SAVING MODELS{' (' + suffix.replace('_', ' ').strip() + ')' if suffix else ''}")
    print(f"{'='*80}")
    
    # Save feature list
    feature_path = os.path.join(output_dir, f'features{suffix}.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_list, f, indent=2)
    print(f"  Saved feature list: {feature_path}")
    
    # Save score model
    if results.get('score') and results['score'] is not None:
        model_path = os.path.join(output_dir, f'score_model{suffix}.json')
        results['score']['model'].save_model(model_path)
        print(f"  Saved score model: {model_path}")
    
    # Save margin model (for spread/moneyline)
    if results.get('margin') and results['margin'] is not None:
        model_path = os.path.join(output_dir, f'margin_model{suffix}.json')
        results['margin']['model'].save_model(model_path)
        print(f"  Saved margin model: {model_path}")
    
    # Save total model
    if results.get('total') and results['total'] is not None:
        model_path = os.path.join(output_dir, f'total_model{suffix}.json')
        results['total']['model'].save_model(model_path)
        print(f"  Saved total model: {model_path}")
    
    # Save model metadata
    metadata = {
        'features': feature_list,
        'metrics': {}
    }
    
    if results.get('score') and results['score'] is not None:
        if 'test_metrics' in results['score']:
            metadata['metrics']['score'] = {
                'mae': results['score']['test_metrics']['mae'],
                'rmse': results['score']['test_metrics']['rmse'],
                'correlation': results['score']['test_metrics']['correlation']
            }
        else:
            metadata['metrics']['score'] = {
                'mae': results['score']['mae'],
                'correlation': results['score']['correlation']
            }
    
    if results.get('margin') and results['margin'] is not None:
        if 'test_metrics' in results['margin']:
            metadata['metrics']['margin'] = {
                'mae': results['margin']['test_metrics']['mae'],
                'rmse': results['margin']['test_metrics']['rmse'],
                'correlation': results['margin']['test_metrics']['correlation']
            }
        else:
            metadata['metrics']['margin'] = {
                'mae': results['margin']['mae'],
                'correlation': results['margin']['correlation'],
                'win_accuracy': results['margin'].get('win_accuracy')
            }
    
    if results.get('total') and results['total'] is not None:
        metadata['metrics']['total'] = {
            'mae': results['total']['mae'],
            'vegas_mae': results['total'].get('vegas_mae'),
            'ou_accuracy': results['total'].get('ou_accuracy'),
            'correlation': results['total'].get('correlation')
        }
    
    if results.get('moneyline') and results['moneyline'] is not None:
        metadata['metrics']['moneyline'] = {
            'win_accuracy': results['moneyline']['win_accuracy'],
            'favorite_accuracy': results['moneyline']['favorite_accuracy']
        }
    
    metadata_path = os.path.join(output_dir, f'model_metadata{suffix}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    print(f"\n  Models saved to: {output_dir}")
    print("  Load with:")
    print("    model = xgb.XGBRegressor()")
    print(f"    model.load_model('margin_model{suffix}.json')")


def main():
    parser = argparse.ArgumentParser(description='NCAAM Model v2')
    parser.add_argument('--bart-games', default='../../processed/base_model_game_data_with_rolling.csv')
    parser.add_argument('--bart-season', default='../../processed/ncaam_team_data_final.csv')
    parser.add_argument('--espn-team', default='../../raw/team_stats.csv')
    parser.add_argument('--espn-player', default='../../raw/player_stats.csv')
    parser.add_argument('--holdout-season', default='2025_26', help='Season to hold out for testing (e.g., 2025_26)')
    parser.add_argument('--output-dir', default='.', help='Directory to save trained models (JSON format)')
    
    args = parser.parse_args()
    
    # Load raw data for diagnostics
    bart_games_raw = pd.read_csv(args.bart_games)
    espn_team_raw = pd.read_csv(args.espn_team)
    
    # Build full dataset with all features
    df = build_full_dataset(
        args.bart_games, args.bart_season, args.espn_team, args.espn_player
    )
    
    # Get feature list
    features = select_features()
    print(f"\n  Selected {len(features)} features for modeling")
    
    # Check what we have
    available, missing = check_feature_availability(df, features)
    print(f"  Available: {len(available)}, Missing: {len(missing)}")
    
    # Vegas data diagnostic
    diagnose_vegas_data(df, bart_games_raw, espn_team_raw)
    
    # Run models for each target
    results = {}
    
    # 1. Score prediction (skip the invalid Vegas comparison)
    if 'target_score' in df.columns:
        results['score'] = run_model_comparison(
            df, features, 'target_score', 'Team Score Prediction',
            vegas_col=None  # Don't compare to total_line, it's not comparable
        )
    
    # 2. Margin prediction (for spread betting)
    if 'target_margin' in df.columns:
        results['margin'] = run_model_comparison(
            df, features, 'target_margin', 'Margin Prediction (Spread)',
            vegas_col='pregame_spread_line'
        )
        
        # Spread betting analysis
        if results['margin']:
            run_spread_betting_analysis(df, results['margin'], features)
        
        # Moneyline analysis
        if results['margin']:
            results['moneyline'] = run_moneyline_analysis(df, results['margin'])
    
    # 3. Total prediction
    results['total'] = run_total_betting_analysis(df, features)
    
    # 4. Held-out season validation (won't work well without full Vegas data)
    run_holdout_season_validation(df, features, args.holdout_season)
    
    # 5. Temporal validation (proper out-of-sample test with Vegas data)
    temporal_results = run_temporal_validation(df, features, test_days=6)
    
    # 6. Train NO-VEGAS models on full dataset
    features_no_vegas = select_features_no_vegas()
    no_vegas_results, no_vegas_available = train_no_vegas_models(df, features_no_vegas)
    
    # Save models
    if args.output_dir:
        # Save Vegas models
        save_models(results, available, args.output_dir, suffix='')
        # Save no-Vegas models
        save_models(no_vegas_results, no_vegas_available, args.output_dir, suffix='_no_vegas')
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print("\n  Baseline Comparison (from previous model):")
    print("    Previous Spread MAE: 8.83 points")
    print("    Previous Total MAE:  13.81 points")
    print("    Vegas Spread MAE:    8.73 points")
    print("    Vegas Total MAE:     13.54 points")
    
    print("\n  WITH VEGAS FEATURES (random split, ~1200 games):")
    if results.get('margin') and results['margin'] is not None:
        print(f"    Spread MAE: {results['margin']['test_metrics']['mae']:.2f} points")
    if results.get('total') and results['total'] is not None:
        print(f"    Total MAE:  {results['total']['mae']:.2f} points")
    if results.get('moneyline') and results['moneyline'] is not None:
        print(f"    Win Prediction: {100*results['moneyline']['win_accuracy']:.1f}%")
    
    if temporal_results:
        print("\n  WITH VEGAS - Temporal Validation (true out-of-sample):")
        if temporal_results.get('margin'):
            print(f"    Spread MAE: {temporal_results['margin']['mae']:.2f} (Vegas: {temporal_results['margin']['vegas_mae']:.2f})")
            print(f"    Win Accuracy: {100*temporal_results['margin']['win_accuracy']:.1f}%")
            print(f"    Cover Accuracy: {100*temporal_results['margin']['cover_accuracy']:.1f}%")
        if temporal_results.get('total'):
            print(f"    Total MAE: {temporal_results['total']['mae']:.2f} (Vegas: {temporal_results['total']['vegas_mae']:.2f})")
            print(f"    O/U Accuracy: {100*temporal_results['total']['ou_accuracy']:.1f}%")
    
    print("\n  NO VEGAS FEATURES (full dataset, ~9000 games):")
    if no_vegas_results.get('margin'):
        print(f"    Spread MAE: {no_vegas_results['margin']['mae']:.2f} points")
        print(f"    Win Accuracy: {100*no_vegas_results['margin']['win_accuracy']:.1f}%")
    if no_vegas_results.get('total'):
        print(f"    Total MAE:  {no_vegas_results['total']['mae']:.2f} points")
    
    print("\n  Models saved:")
    print("    With Vegas:    margin_model.json, total_model.json, score_model.json")
    print("    Without Vegas: margin_model_no_vegas.json, total_model_no_vegas.json, score_model_no_vegas.json")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()