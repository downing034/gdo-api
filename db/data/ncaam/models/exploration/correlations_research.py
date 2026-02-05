"""
NCAAM Phase 0: Exhaustive Correlation Analysis v2

This script incorporates basketball analytics research to test comprehensive derived features:

1. Four Factors matchups (eFG%, TOV%, OReb%, FTR - offense vs corresponding defense)
2. Expected points formula: (team_adjO * opp_adjD / league_avg) * expected_possessions
3. Tempo-adjusted predictions
4. Form/momentum (rolling vs season = trending up or down)
5. Pace mismatches and their effects
6. Strength of schedule adjustments
7. Style matchups (3PT-heavy vs 3PT defense, etc.)

Key principle: A stat is only valid for prediction if you know it BEFORE the game starts.

Usage:
    python correlations_research.py \
        --bart-games /path/to/base_model_game_data_with_rolling.csv \
        --bart-season /path/to/ncaam_team_data_final.csv \
        --espn-team /path/to/team_stats.csv \
        --espn-player /path/to/player_stats.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import argparse
import warnings
warnings.filterwarnings('ignore')

# League average constants (approximate D1 averages)
LEAGUE_AVG_TEMPO = 68.0  # possessions per game
LEAGUE_AVG_EFFICIENCY = 100.0  # points per 100 possessions


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


def calculate_espn_rolling_stats(espn_team):
    """Calculate rolling averages for ESPN team stats."""
    print("Calculating ESPN rolling averages...")
    
    espn_team = espn_team.copy()
    espn_team['game_date'] = pd.to_datetime(espn_team['game_date'])
    espn_team = espn_team.sort_values(['team', 'game_date'])
    
    rolling_cols = [
        'score', 'opponent_score', 'field_goals_made', 'field_goals_attempted',
        'three_pointers_made', 'three_pointers_attempted',
        'free_throws_made', 'free_throws_attempted',
        'offensive_rebounds', 'defensive_rebounds',
        'assists', 'steals', 'blocks', 'turnovers', 'fouls',
        'points_off_turnovers', 'fast_break_points', 'points_in_paint'
    ]
    
    rolling_cols = [c for c in rolling_cols if c in espn_team.columns]
    
    for col in rolling_cols:
        espn_team[f'{col}_rolling_5'] = espn_team.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        espn_team[f'{col}_rolling_10'] = espn_team.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
        )
    
    # Derived rolling stats
    espn_team['efg_pct_rolling_5'] = (
        espn_team['field_goals_made_rolling_5'] + 0.5 * espn_team['three_pointers_made_rolling_5']
    ) / espn_team['field_goals_attempted_rolling_5'].replace(0, np.nan)
    
    espn_team['efg_pct_rolling_10'] = (
        espn_team['field_goals_made_rolling_10'] + 0.5 * espn_team['three_pointers_made_rolling_10']
    ) / espn_team['field_goals_attempted_rolling_10'].replace(0, np.nan)
    
    espn_team['three_pt_pct_rolling_5'] = (
        espn_team['three_pointers_made_rolling_5'] / 
        espn_team['three_pointers_attempted_rolling_5'].replace(0, np.nan)
    )
    espn_team['three_pt_pct_rolling_10'] = (
        espn_team['three_pointers_made_rolling_10'] / 
        espn_team['three_pointers_attempted_rolling_10'].replace(0, np.nan)
    )
    
    espn_team['three_pt_rate_rolling_5'] = (
        espn_team['three_pointers_attempted_rolling_5'] / 
        espn_team['field_goals_attempted_rolling_5'].replace(0, np.nan)
    )
    espn_team['three_pt_rate_rolling_10'] = (
        espn_team['three_pointers_attempted_rolling_10'] / 
        espn_team['field_goals_attempted_rolling_10'].replace(0, np.nan)
    )
    
    espn_team['ft_pct_rolling_5'] = (
        espn_team['free_throws_made_rolling_5'] / 
        espn_team['free_throws_attempted_rolling_5'].replace(0, np.nan)
    )
    espn_team['ft_pct_rolling_10'] = (
        espn_team['free_throws_made_rolling_10'] / 
        espn_team['free_throws_attempted_rolling_10'].replace(0, np.nan)
    )
    
    espn_team['ast_to_tov_rolling_5'] = (
        espn_team['assists_rolling_5'] / 
        espn_team['turnovers_rolling_5'].replace(0, np.nan)
    )
    espn_team['ast_to_tov_rolling_10'] = (
        espn_team['assists_rolling_10'] / 
        espn_team['turnovers_rolling_10'].replace(0, np.nan)
    )
    
    print(f"  Created {len([c for c in espn_team.columns if 'rolling' in c])} rolling stat columns")
    
    return espn_team


def prepare_season_data(bart_season):
    """Prepare season-level team data for joining."""
    print("Preparing season data...")
    
    bart_season = bart_season.copy()
    
    rename_map = {
        'Team_Code': 'team_code',
        'Adj. Off. Eff': 'season_adjO',
        'Adj. Off. Eff Rank': 'season_adjO_rank',
        'Adj. Def. Eff': 'season_adjD',
        'Adj. Def. Eff Rank': 'season_adjD_rank',
        'Barthag': 'season_barthag',
        'Wins': 'season_wins',
        'Games': 'season_games',
        'Eff. FG% Off': 'season_efg_off',
        'Eff. FG% Off Rank': 'season_efg_off_rank',
        'Eff. FG% Def': 'season_efg_def',
        'Eff. FG% Def Rank': 'season_efg_def_rank',
        'FT Rate Off': 'season_ftr_off',
        'FT Rate Off Rank': 'season_ftr_off_rank',
        'FT Rate Def': 'season_ftr_def',
        'FT Rate Def Rank': 'season_ftr_def_rank',
        'Turnover% Off': 'season_tov_off',
        'Turnover% Off Rank': 'season_tov_off_rank',
        'Turnover% Def': 'season_tov_def',
        'Turnover% Def Rank': 'season_tov_def_rank',
        'Off. Reb%': 'season_oreb_pct',
        'Off. Reb% Rank': 'season_oreb_rank',
        'Def. Reb%': 'season_dreb_pct',
        'Def. Reb% Rank': 'season_dreb_rank',
        'Raw Tempo': 'season_tempo_raw',
        '2P% Off': 'season_2pt_off',
        '2P% Off Rank': 'season_2pt_off_rank',
        '2P% Def': 'season_2pt_def',
        '2P% Def Rank': 'season_2pt_def_rank',
        '3P% Off': 'season_3pt_off',
        '3P% Off Rank': 'season_3pt_off_rank',
        '3P% Def': 'season_3pt_def',
        '3P% Def Rank': 'season_3pt_def_rank',
        'Block% Off': 'season_blk_off',
        'Block% Def': 'season_blk_def',
        'Assist% Off': 'season_ast_off',
        'Assist% Def': 'season_ast_def',
        '3P Rate Off': 'season_3pt_rate_off',
        '3P Rate Def': 'season_3pt_rate_def',
        'Adj. Tempo': 'season_tempo_adj',
        'Avg Height': 'season_avg_height',
        'Eff. Height': 'season_eff_height',
        'Experience': 'season_experience',
        'Talent': 'season_talent',
        'FT% Off': 'season_ft_pct_off',
        'FT% Off Rank': 'season_ft_pct_off_rank',
        'FT% Def': 'season_ft_pct_def',
        'FT% Def Rank': 'season_ft_pct_def_rank',
        'PPP Off': 'season_ppp_off',
        'PPP Def': 'season_ppp_def',
        'Elite SOS': 'season_elite_sos',
        'Block% Def Rank': 'season_blk_def_rank',
        'Block% Off Rank': 'season_blk_off_rank',
        'Assist% Off Rank': 'season_ast_off_rank',
        'Assist% Def Rank': 'season_ast_def_rank',
        '3P Rate Off Rank': 'season_3pt_rate_off_rank',
        '3P Rate Def Rank': 'season_3pt_rate_def_rank',
    }
    
    bart_season = bart_season.rename(columns=rename_map)
    
    keep_cols = ['team_code'] + [v for v in rename_map.values() if v != 'team_code']
    keep_cols = [c for c in keep_cols if c in bart_season.columns]
    bart_season = bart_season[keep_cols]
    
    print(f"  Prepared {len(bart_season.columns) - 1} season stat columns")
    
    return bart_season


def calculate_espn_team_derived(df):
    """Calculate derived stats from ESPN team box scores (post-game)."""
    df = df.copy()
    
    df['efg_pct'] = (df['field_goals_made'] + 0.5 * df['three_pointers_made']) / df['field_goals_attempted'].replace(0, np.nan)
    df['true_shooting'] = df['score'] / (2 * (df['field_goals_attempted'] + 0.44 * df['free_throws_attempted'])).replace(0, np.nan)
    df['three_pt_rate'] = df['three_pointers_attempted'] / df['field_goals_attempted'].replace(0, np.nan)
    df['ft_rate'] = df['free_throws_attempted'] / df['field_goals_attempted'].replace(0, np.nan)
    df['two_pt_made'] = df['field_goals_made'] - df['three_pointers_made']
    df['two_pt_att'] = df['field_goals_attempted'] - df['three_pointers_attempted']
    df['two_pt_pct'] = df['two_pt_made'] / df['two_pt_att'].replace(0, np.nan)
    df['fg_pct_calc'] = df['field_goals_made'] / df['field_goals_attempted'].replace(0, np.nan)
    df['three_pt_pct_calc'] = df['three_pointers_made'] / df['three_pointers_attempted'].replace(0, np.nan)
    df['ft_pct_calc'] = df['free_throws_made'] / df['free_throws_attempted'].replace(0, np.nan)
    df['total_rebounds'] = df['offensive_rebounds'] + df['defensive_rebounds']
    df['ast_to_tov'] = df['assists'] / df['turnovers'].replace(0, np.nan)
    df['stocks'] = df['steals'] + df['blocks']
    
    return df


def calculate_player_derived(df):
    """Calculate per-minute and efficiency stats for players."""
    df = df.copy()
    min_minutes = 1
    df_valid = df[df['minutes_played'] >= min_minutes].copy()
    
    df_valid['pts_per_40'] = (df_valid['points'] / df_valid['minutes_played']) * 40
    df_valid['reb_per_40'] = ((df_valid['offensive_rebounds'] + df_valid['defensive_rebounds']) / df_valid['minutes_played']) * 40
    df_valid['ast_per_40'] = (df_valid['assists'] / df_valid['minutes_played']) * 40
    df_valid['stl_per_40'] = (df_valid['steals'] / df_valid['minutes_played']) * 40
    df_valid['blk_per_40'] = (df_valid['blocks'] / df_valid['minutes_played']) * 40
    df_valid['tov_per_40'] = (df_valid['turnovers'] / df_valid['minutes_played']) * 40
    df_valid['stocks_per_40'] = ((df_valid['steals'] + df_valid['blocks']) / df_valid['minutes_played']) * 40
    df_valid['ast_to_tov'] = df_valid['assists'] / df_valid['turnovers'].replace(0, np.nan)
    df_valid['usage'] = (df_valid['field_goals_attempted'] + 0.44 * df_valid['free_throws_attempted'] + df_valid['turnovers']) / df_valid['minutes_played']
    ts_denom = 2 * (df_valid['field_goals_attempted'] + 0.44 * df_valid['free_throws_attempted'])
    df_valid['player_ts'] = df_valid['points'] / ts_denom.replace(0, np.nan)
    
    return df_valid


def aggregate_player_to_team(player_df, espn_team):
    """Aggregate player stats to team level for each game."""
    print("Aggregating player stats to team level...")
    
    player_df = calculate_player_derived(player_df)
    aggregated = []
    
    for (game_id, team), group in player_df.groupby(['game_id', 'team']):
        group = group.sort_values('minutes_played', ascending=False)
        team_total_pts = group['points'].sum()
        team_total_min = group['minutes_played'].sum()
        
        row = {'game_id': game_id, 'team': team}
        
        if len(group) >= 1:
            top = group.iloc[0]
            row['top_player_pts_per_40'] = top.get('pts_per_40', np.nan)
            row['top_player_usage'] = top.get('usage', np.nan)
            row['top_player_ts'] = top.get('player_ts', np.nan)
            row['top_player_minutes'] = top['minutes_played']
            row['top_player_pts'] = top['points']
        
        if len(group) >= 3:
            top3 = group.iloc[:3]
            row['top3_avg_pts_per_40'] = top3['pts_per_40'].mean()
            row['top3_avg_usage'] = top3['usage'].mean()
            row['top3_total_pts'] = top3['points'].sum()
            row['top3_total_min'] = top3['minutes_played'].sum()
        
        if len(group) >= 5:
            starters = group.iloc[:5]
            row['starter_avg_pts_per_40'] = starters['pts_per_40'].mean()
            row['starter_avg_ast_to_tov'] = starters['ast_to_tov'].mean()
            row['starter_avg_stocks_per_40'] = starters['stocks_per_40'].mean()
            row['starter_total_min'] = starters['minutes_played'].sum()
            
            bench = group.iloc[5:]
            if len(bench) > 0:
                bench_pts = bench['points'].sum()
                bench_min = bench['minutes_played'].sum()
                row['bench_pts'] = bench_pts
                row['bench_min'] = bench_min
                row['bench_pts_per_min'] = bench_pts / bench_min if bench_min > 0 else 0
            else:
                row['bench_pts'] = 0
                row['bench_min'] = 0
                row['bench_pts_per_min'] = 0
        
        if team_total_pts > 0 and len(group) >= 1:
            row['scoring_concentration'] = group.iloc[0]['points'] / team_total_pts
        if team_total_min > 0 and len(group) >= 1:
            row['minutes_concentration'] = group.iloc[0]['minutes_played'] / team_total_min
        
        aggregated.append(row)
    
    agg_df = pd.DataFrame(aggregated)
    print(f"  Aggregated {len(agg_df)} team-game rows from player data")
    
    return agg_df


def calculate_player_rolling_stats(player_agg, espn_team):
    """Calculate rolling averages for player aggregate stats."""
    print("Calculating player aggregate rolling averages...")
    
    player_agg = player_agg.merge(
        espn_team[['game_id', 'team', 'game_date']],
        on=['game_id', 'team'],
        how='left'
    )
    
    player_agg['game_date'] = pd.to_datetime(player_agg['game_date'])
    player_agg = player_agg.sort_values(['team', 'game_date'])
    
    rolling_cols = [
        'top_player_pts_per_40', 'top_player_usage', 'top_player_ts',
        'top3_avg_pts_per_40', 'top3_avg_usage',
        'starter_avg_pts_per_40', 'starter_avg_ast_to_tov', 'starter_avg_stocks_per_40',
        'bench_pts_per_min', 'scoring_concentration', 'minutes_concentration'
    ]
    
    rolling_cols = [c for c in rolling_cols if c in player_agg.columns]
    
    for col in rolling_cols:
        player_agg[f'{col}_rolling_5'] = player_agg.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        player_agg[f'{col}_rolling_10'] = player_agg.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
        )
    
    print(f"  Created {len([c for c in player_agg.columns if 'rolling' in c])} rolling columns")
    
    return player_agg


def calculate_derived_features(df):
    """
    Calculate advanced derived features based on basketball analytics research.
    
    Features based on:
    - Dean Oliver's Four Factors
    - KenPom efficiency methodology
    - Tempo/pace adjustments
    - Matchup-specific combinations
    - Pythagorean expectation
    - Variance/consistency metrics
    - Logarithmic transforms
    - Polynomial features
    - Interaction effects
    """
    print("\nCalculating advanced derived features...")
    df = df.copy()
    
    # =========================================================================
    # HELPER: Safe division
    # =========================================================================
    def safe_div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b != 0, a / b, np.nan)
    
    # =========================================================================
    # 1. EXPECTED POINTS FORMULA (KenPom-style)
    # Expected points = (team_adjO * opp_adjD / league_avg) * (expected_possessions / 100)
    # =========================================================================
    
    # Expected tempo when these two teams play (additive model per KenPom)
    # expected_tempo = league_avg + (team_tempo - league_avg) + (opp_tempo - league_avg)
    if 'pregame_team_season_tempo_adj' in df.columns and 'pregame_opp_season_tempo_adj' in df.columns:
        df['derived_expected_tempo'] = (
            LEAGUE_AVG_TEMPO + 
            (df['pregame_team_season_tempo_adj'] - LEAGUE_AVG_TEMPO) + 
            (df['pregame_opp_season_tempo_adj'] - LEAGUE_AVG_TEMPO)
        )
    
    # Expected offensive efficiency when team plays this opponent (additive per KenPom research)
    # team_expected_eff = league_avg + (team_adjO - league_avg) + (opp_adjD - league_avg)
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_expected_off_eff'] = (
            LEAGUE_AVG_EFFICIENCY + 
            (df['pregame_team_season_adjO'] - LEAGUE_AVG_EFFICIENCY) + 
            (df['pregame_opp_season_adjD'] - LEAGUE_AVG_EFFICIENCY)
        )
        
        # Expected points = expected_eff * (expected_tempo / 100)
        if 'derived_expected_tempo' in df.columns:
            df['derived_expected_points'] = (
                df['derived_expected_off_eff'] * df['derived_expected_tempo'] / 100
            )
    
    # =========================================================================
    # 2. FOUR FACTORS MATCHUPS (offense vs corresponding defense)
    # =========================================================================
    
    # eFG% matchup: team offensive eFG% vs opponent defensive eFG%
    if 'pregame_team_season_efg_off' in df.columns and 'pregame_opp_season_efg_def' in df.columns:
        df['derived_efg_matchup_diff'] = df['pregame_team_season_efg_off'] - df['pregame_opp_season_efg_def']
        df['derived_efg_matchup_ratio'] = df['pregame_team_season_efg_off'] / df['pregame_opp_season_efg_def'].replace(0, np.nan)
    
    # Turnover% matchup: team offensive TOV% vs opponent forced TOV%
    if 'pregame_team_season_tov_off' in df.columns and 'pregame_opp_season_tov_def' in df.columns:
        df['derived_tov_matchup_diff'] = df['pregame_team_season_tov_off'] - df['pregame_opp_season_tov_def']
        # Lower is better for offense, higher is better for defense forcing TOs
        # Negative diff = team turns it over less than opponent forces
    
    # Rebounding matchup: team OReb% vs opponent DReb%
    if 'pregame_team_season_oreb_pct' in df.columns and 'pregame_opp_season_dreb_pct' in df.columns:
        df['derived_oreb_matchup_diff'] = df['pregame_team_season_oreb_pct'] - df['pregame_opp_season_dreb_pct']
        # Positive = team gets more offensive rebounds than opponent prevents
    
    # FT Rate matchup: team FTR vs opponent FTR allowed
    if 'pregame_team_season_ftr_off' in df.columns and 'pregame_opp_season_ftr_def' in df.columns:
        df['derived_ftr_matchup_diff'] = df['pregame_team_season_ftr_off'] - df['pregame_opp_season_ftr_def']
    
    # =========================================================================
    # 3. SHOT TYPE MATCHUPS
    # =========================================================================
    
    # 3PT shooting vs 3PT defense
    if 'pregame_team_season_3pt_off' in df.columns and 'pregame_opp_season_3pt_def' in df.columns:
        df['derived_3pt_matchup_diff'] = df['pregame_team_season_3pt_off'] - df['pregame_opp_season_3pt_def']
        df['derived_3pt_matchup_ratio'] = df['pregame_team_season_3pt_off'] / df['pregame_opp_season_3pt_def'].replace(0, np.nan)
    
    # 2PT shooting vs 2PT defense
    if 'pregame_team_season_2pt_off' in df.columns and 'pregame_opp_season_2pt_def' in df.columns:
        df['derived_2pt_matchup_diff'] = df['pregame_team_season_2pt_off'] - df['pregame_opp_season_2pt_def']
        df['derived_2pt_matchup_ratio'] = df['pregame_team_season_2pt_off'] / df['pregame_opp_season_2pt_def'].replace(0, np.nan)
    
    # 3PT rate vs 3PT rate defense (team shoots 3s vs opponent allows 3s)
    if 'pregame_team_season_3pt_rate_off' in df.columns and 'pregame_opp_season_3pt_rate_def' in df.columns:
        df['derived_3pt_rate_matchup'] = df['pregame_team_season_3pt_rate_off'] - df['pregame_opp_season_3pt_rate_def']
    
    # =========================================================================
    # 4. EFFICIENCY DIFFERENTIALS
    # =========================================================================
    
    # Net efficiency differential (team's net vs opponent's net)
    if 'pregame_team_season_adjO' in df.columns and 'pregame_team_season_adjD' in df.columns:
        df['derived_team_net_eff'] = df['pregame_team_season_adjO'] - df['pregame_team_season_adjD']
    
    if 'pregame_opp_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_opp_net_eff'] = df['pregame_opp_season_adjO'] - df['pregame_opp_season_adjD']
    
    if 'derived_team_net_eff' in df.columns and 'derived_opp_net_eff' in df.columns:
        df['derived_net_eff_diff'] = df['derived_team_net_eff'] - df['derived_opp_net_eff']
    
    # AdjO - opponent's AdjD (how much better/worse than opponent allows)
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_adjO_vs_adjD'] = df['pregame_team_season_adjO'] - df['pregame_opp_season_adjD']
    
    # =========================================================================
    # 5. TEMPO/PACE FEATURES
    # =========================================================================
    
    # Pace mismatch (how different are the two teams' preferred pace)
    if 'pregame_team_season_tempo_adj' in df.columns and 'pregame_opp_season_tempo_adj' in df.columns:
        df['derived_pace_mismatch'] = abs(df['pregame_team_season_tempo_adj'] - df['pregame_opp_season_tempo_adj'])
        df['derived_pace_diff'] = df['pregame_team_season_tempo_adj'] - df['pregame_opp_season_tempo_adj']
        
        # Combined pace (both fast = high, both slow = low)
        df['derived_combined_tempo'] = df['pregame_team_season_tempo_adj'] + df['pregame_opp_season_tempo_adj']
    
    # Efficiency * Tempo interactions (high efficiency at fast pace = more points)
    if 'pregame_team_season_adjO' in df.columns and 'pregame_team_season_tempo_adj' in df.columns:
        df['derived_adjO_x_tempo'] = df['pregame_team_season_adjO'] * df['pregame_team_season_tempo_adj'] / 100
    
    # =========================================================================
    # 6. FORM/MOMENTUM (rolling vs season = trending)
    # =========================================================================
    
    # Offensive form: rolling AdjO vs season AdjO (positive = trending up)
    if 'pregame_team_AdjO_rolling_10' in df.columns and 'pregame_team_season_adjO' in df.columns:
        df['derived_adjO_form'] = df['pregame_team_AdjO_rolling_10'] - df['pregame_team_season_adjO']
    
    if 'pregame_team_AdjO_rolling_5' in df.columns and 'pregame_team_AdjO_rolling_10' in df.columns:
        df['derived_adjO_recent_trend'] = df['pregame_team_AdjO_rolling_5'] - df['pregame_team_AdjO_rolling_10']
    
    # eFG form
    if 'pregame_team_eFG_off_rolling_10' in df.columns and 'pregame_team_season_efg_off' in df.columns:
        df['derived_efg_form'] = df['pregame_team_eFG_off_rolling_10'] - df['pregame_team_season_efg_off']
    
    # Scoring form
    if 'pregame_team_score_rolling_10' in df.columns and 'pregame_team_score_rolling_5' in df.columns:
        df['derived_scoring_trend'] = df['pregame_team_score_rolling_5'] - df['pregame_team_score_rolling_10']
    
    # =========================================================================
    # 7. REST AND CONTEXT
    # =========================================================================
    
    if 'pregame_team_days_rest' in df.columns and 'pregame_opp_days_rest' in df.columns:
        df['derived_rest_advantage'] = df['pregame_team_days_rest'] - df['pregame_opp_days_rest']
    
    # =========================================================================
    # 8. STRENGTH OF SCHEDULE CONTEXT
    # =========================================================================
    
    if 'pregame_team_sos' in df.columns and 'pregame_opp_sos' in df.columns:
        df['derived_sos_diff'] = df['pregame_team_sos'] - df['pregame_opp_sos']
    
    # =========================================================================
    # 9. STYLE INDICATORS (Boolean-like features)
    # =========================================================================
    
    # Is team 3PT-dependent? (high 3PT rate)
    if 'pregame_team_season_3pt_rate_off' in df.columns:
        median_3pt_rate = df['pregame_team_season_3pt_rate_off'].median()
        df['derived_team_3pt_heavy'] = (df['pregame_team_season_3pt_rate_off'] > median_3pt_rate).astype(int)
    
    # Does opponent defend 3PT well?
    if 'pregame_opp_season_3pt_def' in df.columns:
        median_3pt_def = df['pregame_opp_season_3pt_def'].median()
        df['derived_opp_good_3pt_def'] = (df['pregame_opp_season_3pt_def'] < median_3pt_def).astype(int)
    
    # 3PT-dependent team vs good 3PT defense (potential trouble)
    if 'derived_team_3pt_heavy' in df.columns and 'derived_opp_good_3pt_def' in df.columns:
        df['derived_3pt_trouble_matchup'] = df['derived_team_3pt_heavy'] * df['derived_opp_good_3pt_def']
    
    # =========================================================================
    # 10. BLOCK/SHOT CONTEST MATCHUPS
    # =========================================================================
    
    if 'pregame_team_season_blk_off' in df.columns and 'pregame_opp_season_blk_def' in df.columns:
        df['derived_blk_matchup'] = df['pregame_team_season_blk_off'] - df['pregame_opp_season_blk_def']
    
    # =========================================================================
    # 11. ASSIST RATE MATCHUPS
    # =========================================================================
    
    if 'pregame_team_season_ast_off' in df.columns and 'pregame_opp_season_ast_def' in df.columns:
        df['derived_ast_matchup'] = df['pregame_team_season_ast_off'] - df['pregame_opp_season_ast_def']
    
    # =========================================================================
    # 12. COMPOSITE SCORES
    # =========================================================================
    
    # Four factors composite (weighted by Oliver's weights: 40/25/20/15)
    if all(col in df.columns for col in ['derived_efg_matchup_diff', 'derived_tov_matchup_diff', 
                                          'derived_oreb_matchup_diff', 'derived_ftr_matchup_diff']):
        df['derived_four_factors_composite'] = (
            0.40 * df['derived_efg_matchup_diff'] * 100 +  # Scale eFG diff to similar magnitude
            0.25 * (-df['derived_tov_matchup_diff']) +  # Negative because lower TO is better
            0.20 * df['derived_oreb_matchup_diff'] +
            0.15 * df['derived_ftr_matchup_diff']
        )
    
    # =========================================================================
    # 13. PYTHAGOREAN EXPECTATION (KenPom style)
    # =========================================================================
    
    # Pythagorean win expectation: AdjO^11.5 / (AdjO^11.5 + AdjD^11.5)
    if 'pregame_team_season_adjO' in df.columns and 'pregame_team_season_adjD' in df.columns:
        adjO = df['pregame_team_season_adjO']
        adjD = df['pregame_team_season_adjD']
        df['derived_team_pyth'] = (adjO ** 11.5) / ((adjO ** 11.5) + (adjD ** 11.5))
        
    if 'pregame_opp_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        opp_adjO = df['pregame_opp_season_adjO']
        opp_adjD = df['pregame_opp_season_adjD']
        df['derived_opp_pyth'] = (opp_adjO ** 11.5) / ((opp_adjO ** 11.5) + (opp_adjD ** 11.5))
        
    if 'derived_team_pyth' in df.columns and 'derived_opp_pyth' in df.columns:
        df['derived_pyth_diff'] = df['derived_team_pyth'] - df['derived_opp_pyth']
        # Log5 formula for head-to-head probability
        pA = df['derived_team_pyth']
        pB = df['derived_opp_pyth']
        df['derived_log5_prob'] = (pA - pA * pB) / (pA + pB - 2 * pA * pB)
    
    # =========================================================================
    # 14. EXPECTED TOTAL POINTS (for total prediction)
    # =========================================================================
    
    if 'derived_expected_points' in df.columns:
        # Team's expected + opponent's expected (need to calc opponent's expected)
        if 'pregame_opp_season_adjO' in df.columns and 'pregame_team_season_adjD' in df.columns:
            df['derived_opp_expected_off_eff'] = (
                LEAGUE_AVG_EFFICIENCY + 
                (df['pregame_opp_season_adjO'] - LEAGUE_AVG_EFFICIENCY) + 
                (df['pregame_team_season_adjD'] - LEAGUE_AVG_EFFICIENCY)
            )
            if 'derived_expected_tempo' in df.columns:
                df['derived_opp_expected_points'] = (
                    df['derived_opp_expected_off_eff'] * df['derived_expected_tempo'] / 100
                )
                df['derived_expected_total'] = df['derived_expected_points'] + df['derived_opp_expected_points']
                df['derived_expected_spread'] = df['derived_expected_points'] - df['derived_opp_expected_points']
    
    # =========================================================================
    # 15. LOGARITHMIC TRANSFORMS (capture diminishing returns)
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns:
        df['derived_log_adjO'] = np.log(df['pregame_team_season_adjO'])
    if 'pregame_opp_season_adjD' in df.columns:
        df['derived_log_opp_adjD'] = np.log(df['pregame_opp_season_adjD'])
    if 'derived_expected_points' in df.columns:
        df['derived_log_expected_pts'] = np.log(df['derived_expected_points'].clip(lower=1))
    
    # =========================================================================
    # 16. SQUARED/POLYNOMIAL FEATURES
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns:
        df['derived_adjO_squared'] = df['pregame_team_season_adjO'] ** 2
    if 'derived_net_eff_diff' in df.columns:
        df['derived_net_eff_diff_squared'] = df['derived_net_eff_diff'] ** 2
        df['derived_net_eff_diff_sign'] = np.sign(df['derived_net_eff_diff']) * (df['derived_net_eff_diff'] ** 2)
    if 'derived_expected_tempo' in df.columns:
        df['derived_tempo_squared'] = df['derived_expected_tempo'] ** 2
    
    # =========================================================================
    # 17. INTERACTION WITH HOME/AWAY
    # =========================================================================
    
    if 'is_home' in df.columns:
        if 'pregame_team_season_adjO' in df.columns:
            df['derived_adjO_x_home'] = df['pregame_team_season_adjO'] * df['is_home']
        if 'derived_expected_points' in df.columns:
            df['derived_exp_pts_x_home'] = df['derived_expected_points'] * df['is_home']
        if 'pregame_team_season_tempo_adj' in df.columns:
            df['derived_tempo_x_home'] = df['pregame_team_season_tempo_adj'] * df['is_home']
        if 'derived_net_eff_diff' in df.columns:
            df['derived_net_eff_x_home'] = df['derived_net_eff_diff'] * df['is_home']
    
    # =========================================================================
    # 18. ROLLING FORM INTERACTIONS
    # =========================================================================
    
    # Recent form × opponent quality
    if 'pregame_team_AdjO_rolling_5' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_recent_form_x_opp_def'] = df['pregame_team_AdjO_rolling_5'] * df['pregame_opp_season_adjD']
    
    # Form trend × tempo
    if 'derived_adjO_form' in df.columns and 'derived_expected_tempo' in df.columns:
        df['derived_form_x_tempo'] = df['derived_adjO_form'] * df['derived_expected_tempo']
    
    # =========================================================================
    # 19. CONSISTENCY/VARIANCE PROXIES
    # =========================================================================
    
    # Difference between rolling_5 and rolling_10 as volatility proxy
    if 'pregame_team_AdjO_rolling_5' in df.columns and 'pregame_team_AdjO_rolling_10' in df.columns:
        df['derived_adjO_volatility'] = abs(df['pregame_team_AdjO_rolling_5'] - df['pregame_team_AdjO_rolling_10'])
    
    if 'pregame_team_score_rolling_5' in df.columns and 'pregame_team_score_rolling_10' in df.columns:
        df['derived_scoring_volatility'] = abs(df['pregame_team_score_rolling_5'] - df['pregame_team_score_rolling_10'])
    
    # =========================================================================
    # 20. RANKING-BASED FEATURES
    # =========================================================================
    
    if 'pregame_team_season_adjO_rank' in df.columns and 'pregame_opp_season_adjD_rank' in df.columns:
        df['derived_rank_sum'] = df['pregame_team_season_adjO_rank'] + df['pregame_opp_season_adjD_rank']
        df['derived_rank_diff'] = df['pregame_team_season_adjO_rank'] - df['pregame_opp_season_adjD_rank']
        df['derived_rank_product'] = df['pregame_team_season_adjO_rank'] * df['pregame_opp_season_adjD_rank']
    
    # =========================================================================
    # 21. EXPERIENCE AND TALENT INTERACTIONS
    # =========================================================================
    
    if 'pregame_team_season_experience' in df.columns and 'pregame_team_season_talent' in df.columns:
        df['derived_exp_x_talent'] = df['pregame_team_season_experience'] * df['pregame_team_season_talent']
        df['derived_exp_plus_talent'] = df['pregame_team_season_experience'] + df['pregame_team_season_talent']
    
    if 'pregame_team_season_talent' in df.columns and 'pregame_opp_season_talent' in df.columns:
        df['derived_talent_diff'] = df['pregame_team_season_talent'] - df['pregame_opp_season_talent']
        df['derived_talent_ratio'] = safe_div(df['pregame_team_season_talent'], df['pregame_opp_season_talent'])
    
    if 'pregame_team_season_experience' in df.columns and 'pregame_opp_season_experience' in df.columns:
        df['derived_experience_diff'] = df['pregame_team_season_experience'] - df['pregame_opp_season_experience']
    
    # =========================================================================
    # 22. HEIGHT ADVANTAGE
    # =========================================================================
    
    if 'pregame_team_season_avg_height' in df.columns and 'pregame_opp_season_avg_height' in df.columns:
        df['derived_height_diff'] = df['pregame_team_season_avg_height'] - df['pregame_opp_season_avg_height']
        
    if 'pregame_team_season_eff_height' in df.columns and 'pregame_opp_season_eff_height' in df.columns:
        df['derived_eff_height_diff'] = df['pregame_team_season_eff_height'] - df['pregame_opp_season_eff_height']
    
    # Height × rebounding matchup
    if 'derived_height_diff' in df.columns and 'derived_oreb_matchup_diff' in df.columns:
        df['derived_height_x_reb'] = df['derived_height_diff'] * df['derived_oreb_matchup_diff']
    
    # =========================================================================
    # 23. OFFENSIVE STYLE METRICS
    # =========================================================================
    
    # 3PT dependency score (3PT rate × 3PT%)
    if 'pregame_team_season_3pt_rate_off' in df.columns and 'pregame_team_season_3pt_off' in df.columns:
        df['derived_3pt_dependency'] = df['pregame_team_season_3pt_rate_off'] * df['pregame_team_season_3pt_off']
    
    # Paint dependency (2PT% × (1 - 3PT rate))
    if 'pregame_team_season_2pt_off' in df.columns and 'pregame_team_season_3pt_rate_off' in df.columns:
        df['derived_paint_dependency'] = df['pregame_team_season_2pt_off'] * (1 - df['pregame_team_season_3pt_rate_off'])
    
    # FT dependency
    if 'pregame_team_season_ftr_off' in df.columns and 'pregame_team_season_ft_pct_off' in df.columns:
        df['derived_ft_dependency'] = df['pregame_team_season_ftr_off'] * df['pregame_team_season_ft_pct_off']
    
    # =========================================================================
    # 24. DEFENSIVE STYLE METRICS
    # =========================================================================
    
    # How much does opponent force turnovers?
    if 'pregame_opp_season_tov_def' in df.columns and 'pregame_team_season_tov_off' in df.columns:
        df['derived_tov_pressure'] = df['pregame_opp_season_tov_def'] - df['pregame_team_season_tov_off']
    
    # Opponent's rim protection vs team's paint scoring
    if 'pregame_opp_season_blk_def' in df.columns and 'pregame_team_season_2pt_off' in df.columns:
        df['derived_rim_protection_matchup'] = df['pregame_opp_season_blk_def'] * (1 - df['pregame_team_season_3pt_rate_off']) if 'pregame_team_season_3pt_rate_off' in df.columns else df['pregame_opp_season_blk_def']
    
    # =========================================================================
    # 25. GAME CONTEXT INTERACTIONS
    # =========================================================================
    
    # Rest × efficiency
    if 'pregame_team_days_rest' in df.columns and 'pregame_team_season_adjO' in df.columns:
        df['derived_rest_x_adjO'] = df['pregame_team_days_rest'] * df['pregame_team_season_adjO']
    
    # Rest advantage × form
    if 'derived_rest_advantage' in df.columns and 'derived_adjO_form' in df.columns:
        df['derived_rest_x_form'] = df['derived_rest_advantage'] * df['derived_adjO_form']
    
    # =========================================================================
    # 26. CEILING/FLOOR ESTIMATES
    # =========================================================================
    
    # Optimistic estimate (use better of season vs rolling)
    if 'pregame_team_season_adjO' in df.columns and 'pregame_team_AdjO_rolling_10' in df.columns:
        df['derived_adjO_ceiling'] = df[['pregame_team_season_adjO', 'pregame_team_AdjO_rolling_10']].max(axis=1)
        df['derived_adjO_floor'] = df[['pregame_team_season_adjO', 'pregame_team_AdjO_rolling_10']].min(axis=1)
    
    # =========================================================================
    # 27. PACE-ADJUSTED EVERYTHING
    # =========================================================================
    
    # All major stats pace-adjusted
    if 'derived_expected_tempo' in df.columns:
        pace_factor = df['derived_expected_tempo'] / LEAGUE_AVG_TEMPO
        
        if 'pregame_team_season_adjO' in df.columns:
            df['derived_pace_adj_adjO'] = df['pregame_team_season_adjO'] * pace_factor
        if 'derived_efg_matchup_diff' in df.columns:
            df['derived_pace_adj_efg_matchup'] = df['derived_efg_matchup_diff'] * pace_factor
        if 'derived_net_eff_diff' in df.columns:
            df['derived_pace_adj_net_eff'] = df['derived_net_eff_diff'] * pace_factor
    
    # =========================================================================
    # 28. OPPONENT-ADJUSTED ROLLING STATS
    # =========================================================================
    
    # Rolling AdjO adjusted by opponent defensive quality
    if 'pregame_team_AdjO_rolling_10' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_opp_adj_rolling_adjO'] = (
            df['pregame_team_AdjO_rolling_10'] + 
            (LEAGUE_AVG_EFFICIENCY - df['pregame_opp_season_adjD'])
        )
    
    # =========================================================================
    # 29. BLOWOUT POTENTIAL
    # =========================================================================
    
    # Large efficiency gap + fast pace = potential blowout
    if 'derived_net_eff_diff' in df.columns and 'derived_expected_tempo' in df.columns:
        df['derived_blowout_potential'] = abs(df['derived_net_eff_diff']) * df['derived_expected_tempo']
    
    # =========================================================================
    # 30. UPSET POTENTIAL
    # =========================================================================
    
    # Underdog has high variance + pace mismatch favors them
    if 'derived_adjO_volatility' in df.columns and 'derived_pace_diff' in df.columns:
        df['derived_upset_potential'] = df['derived_adjO_volatility'] * abs(df['derived_pace_diff'])
    
    # =========================================================================
    # 31. SHOOTING VOLUME × EFFICIENCY
    # =========================================================================
    
    if 'pregame_team_field_goals_attempted_rolling_10' in df.columns and 'pregame_team_efg_pct_rolling_10' in df.columns:
        df['derived_shot_volume_x_eff'] = df['pregame_team_field_goals_attempted_rolling_10'] * df['pregame_team_efg_pct_rolling_10']
    
    if 'pregame_team_three_pointers_attempted_rolling_10' in df.columns and 'pregame_team_three_pt_pct_rolling_10' in df.columns:
        df['derived_3pt_volume_x_eff'] = df['pregame_team_three_pointers_attempted_rolling_10'] * df['pregame_team_three_pt_pct_rolling_10']
    
    # =========================================================================
    # 32. ASSIST-DEPENDENT VS ISO SCORING
    # =========================================================================
    
    if 'pregame_team_season_ast_off' in df.columns:
        # High assist % = team basketball, low = ISO heavy
        median_ast = df['pregame_team_season_ast_off'].median()
        df['derived_team_basketball'] = (df['pregame_team_season_ast_off'] > median_ast).astype(int)
    
    # =========================================================================
    # 33. DEFENSIVE PRESSURE COMPOSITE
    # =========================================================================
    
    if all(c in df.columns for c in ['pregame_opp_season_tov_def', 'pregame_opp_season_blk_def', 'pregame_opp_season_efg_def']):
        # Normalize each component and combine
        df['derived_def_pressure_composite'] = (
            df['pregame_opp_season_tov_def'] / df['pregame_opp_season_tov_def'].mean() +
            df['pregame_opp_season_blk_def'] / df['pregame_opp_season_blk_def'].mean() +
            (1 - df['pregame_opp_season_efg_def'] / df['pregame_opp_season_efg_def'].mean())
        )
    
    # =========================================================================
    # 34. REBOUNDING BATTLE PREDICTION
    # =========================================================================
    
    if 'pregame_team_season_oreb_pct' in df.columns and 'pregame_opp_season_dreb_pct' in df.columns:
        # Expected OReb% in this game
        df['derived_expected_oreb'] = df['pregame_team_season_oreb_pct'] * (1 - df['pregame_opp_season_dreb_pct'] / 100)
    
    # =========================================================================
    # 35. FREE THROW BATTLE
    # =========================================================================
    
    if 'pregame_team_season_ftr_off' in df.columns and 'pregame_opp_season_ftr_def' in df.columns:
        df['derived_expected_ftr'] = (df['pregame_team_season_ftr_off'] + df['pregame_opp_season_ftr_def']) / 2
        
        if 'pregame_team_season_ft_pct_off' in df.columns:
            df['derived_expected_ft_points'] = df['derived_expected_ftr'] * df['pregame_team_season_ft_pct_off'] * 0.75  # Rough pts from FTs
    
    # =========================================================================
    # 36. SCORING DISTRIBUTION MATCHUPS
    # =========================================================================
    
    if 'pregame_team_scoring_concentration_rolling_10' in df.columns:
        # High concentration = star-dependent
        df['derived_star_dependent'] = df['pregame_team_scoring_concentration_rolling_10']
        
        # If opponent has good perimeter D and team is star-dependent
        if 'pregame_opp_season_efg_def' in df.columns:
            df['derived_star_vs_def'] = df['derived_star_dependent'] * (1 - df['pregame_opp_season_efg_def'])
    
    # =========================================================================
    # 37. BENCH DEPTH IMPACT
    # =========================================================================
    
    if 'pregame_team_bench_pts_per_min_rolling_10' in df.columns:
        # Bench production × expected pace
        if 'derived_expected_tempo' in df.columns:
            df['derived_bench_impact'] = df['pregame_team_bench_pts_per_min_rolling_10'] * df['derived_expected_tempo'] / 10
    
    # =========================================================================
    # 38. INVERSE/RECIPROCAL FEATURES
    # =========================================================================
    
    if 'pregame_opp_season_adjD' in df.columns:
        df['derived_inv_opp_adjD'] = safe_div(1, df['pregame_opp_season_adjD']) * 10000
    
    # =========================================================================
    # 39. NORMALIZED EFFICIENCY GAP
    # =========================================================================
    
    if 'derived_net_eff_diff' in df.columns:
        # Z-score the efficiency difference
        mean_diff = df['derived_net_eff_diff'].mean()
        std_diff = df['derived_net_eff_diff'].std()
        df['derived_net_eff_zscore'] = (df['derived_net_eff_diff'] - mean_diff) / std_diff
    
    # =========================================================================
    # 40. VEGAS COMPARISON FEATURES
    # =========================================================================
    
    if 'pregame_total_line' in df.columns and 'derived_expected_total' in df.columns:
        df['derived_vegas_total_diff'] = df['derived_expected_total'] - df['pregame_total_line']
    
    if 'pregame_spread_line' in df.columns and 'derived_expected_spread' in df.columns:
        df['derived_vegas_spread_diff'] = df['derived_expected_spread'] - (-df['pregame_spread_line'])  # Negate because spread is from other perspective
    
    # =========================================================================
    # 41. POINTS PER POSSESSION MATCHUP (using rolling)
    # =========================================================================
    
    if 'pregame_team_score_rolling_10' in df.columns and 'pregame_team_T_rolling_10' in df.columns:
        df['derived_team_ppp_rolling'] = safe_div(df['pregame_team_score_rolling_10'], df['pregame_team_T_rolling_10']) * 100
    
    if 'pregame_opp_score_rolling_10' in df.columns and 'pregame_opp_T_rolling_10' in df.columns:
        df['derived_opp_ppp_rolling'] = safe_div(df['pregame_opp_score_rolling_10'], df['pregame_opp_T_rolling_10']) * 100
    
    # =========================================================================
    # 42. CLUTCH/CLOSE GAME INDICATORS (using win% vs efficiency gap)
    # =========================================================================
    
    if 'pregame_team_season_wins' in df.columns and 'pregame_team_season_games' in df.columns:
        df['derived_team_win_pct'] = safe_div(df['pregame_team_season_wins'], df['pregame_team_season_games'])
        
        # Luck = actual win% - expected win% (pythagorean)
        if 'derived_team_pyth' in df.columns:
            df['derived_team_luck'] = df['derived_team_win_pct'] - df['derived_team_pyth']
    
    # =========================================================================
    # 43. ABSOLUTE VALUE FEATURES (for spread prediction)
    # =========================================================================
    
    if 'derived_net_eff_diff' in df.columns:
        df['derived_abs_eff_diff'] = abs(df['derived_net_eff_diff'])
    
    if 'derived_talent_diff' in df.columns:
        df['derived_abs_talent_diff'] = abs(df['derived_talent_diff'])
    
    # =========================================================================
    # 44. INTERACTION PRODUCTS - EVERYTHING x EVERYTHING
    # =========================================================================
    
    # Expected points variants
    if 'derived_expected_points' in df.columns:
        if 'is_home' in df.columns:
            df['derived_exp_pts_x_home'] = df['derived_expected_points'] * df['is_home']
        if 'pregame_team_days_rest' in df.columns:
            df['derived_exp_pts_x_rest'] = df['derived_expected_points'] * df['pregame_team_days_rest']
    
    # Net efficiency variants
    if 'derived_net_eff_diff' in df.columns:
        if 'is_home' in df.columns:
            df['derived_net_eff_x_home'] = df['derived_net_eff_diff'] * df['is_home']
        if 'pregame_team_season_tempo_adj' in df.columns:
            df['derived_net_eff_x_tempo'] = df['derived_net_eff_diff'] * df['pregame_team_season_tempo_adj']
    
    # =========================================================================
    # 45. ROLLING STAT BLENDS
    # =========================================================================
    
    if 'pregame_team_AdjO_rolling_5' in df.columns and 'pregame_team_AdjO_rolling_10' in df.columns:
        df['derived_adjO_rolling_blend'] = 0.6 * df['pregame_team_AdjO_rolling_5'] + 0.4 * df['pregame_team_AdjO_rolling_10']
    
    if 'pregame_team_AdjO_rolling_10' in df.columns and 'pregame_team_season_adjO' in df.columns:
        df['derived_adjO_rolling_season_blend'] = 0.5 * df['pregame_team_AdjO_rolling_10'] + 0.5 * df['pregame_team_season_adjO']
    
    # =========================================================================
    # 46. SIMPLE RAW SCORING FEATURES FROM ROLLING
    # =========================================================================
    
    if 'pregame_team_score_rolling_10' in df.columns and 'pregame_opp_score_rolling_10' in df.columns:
        df['derived_score_diff_rolling'] = df['pregame_team_score_rolling_10'] - df['pregame_opp_score_rolling_10']
        df['derived_combined_scoring_rolling'] = df['pregame_team_score_rolling_10'] + df['pregame_opp_score_rolling_10']
    
    # =========================================================================
    # 47. DEFENSIVE MATCHUP FEATURES
    # =========================================================================
    
    if 'pregame_opp_season_adjD' in df.columns and 'pregame_team_season_adjO' in df.columns:
        # Offensive efficiency minus defensive efficiency allowed
        df['derived_off_vs_def_matchup'] = df['pregame_team_season_adjO'] - df['pregame_opp_season_adjD']
    
    if 'pregame_opp_AdjD_rolling_10' in df.columns and 'pregame_team_AdjO_rolling_10' in df.columns:
        df['derived_off_vs_def_matchup_rolling'] = df['pregame_team_AdjO_rolling_10'] - df['pregame_opp_AdjD_rolling_10']
    
    # =========================================================================
    # 48. SIMPLE PRODUCTS THAT MIGHT WORK
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_adjO_adjD_product'] = df['pregame_team_season_adjO'] * df['pregame_opp_season_adjD'] / 10000
    
    if 'pregame_team_season_ppp_off' in df.columns and 'pregame_opp_season_ppp_def' in df.columns:
        df['derived_ppp_product'] = df['pregame_team_season_ppp_off'] * df['pregame_opp_season_ppp_def']
    
    if 'pregame_team_season_efg_off' in df.columns and 'pregame_opp_season_efg_def' in df.columns:
        df['derived_efg_product'] = df['pregame_team_season_efg_off'] * df['pregame_opp_season_efg_def']
    
    # =========================================================================
    # 49. RATIO FEATURES
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_adjO_adjD_ratio'] = safe_div(df['pregame_team_season_adjO'], df['pregame_opp_season_adjD'])
    
    if 'pregame_team_season_tempo_adj' in df.columns and 'pregame_opp_season_tempo_adj' in df.columns:
        df['derived_tempo_ratio'] = safe_div(df['pregame_team_season_tempo_adj'], df['pregame_opp_season_tempo_adj'])
    
    # =========================================================================
    # 50. VEGAS-DERIVED EFFICIENCY ESTIMATE
    # =========================================================================
    
    if 'pregame_total_line' in df.columns and 'pregame_spread_line' in df.columns:
        # From total and spread, estimate team's expected points
        df['derived_vegas_implied_score'] = (df['pregame_total_line'] - df['pregame_spread_line']) / 2
        df['derived_vegas_implied_opp_score'] = (df['pregame_total_line'] + df['pregame_spread_line']) / 2
    
    # =========================================================================
    # 51. CONSISTENCY METRICS
    # =========================================================================
    
    if 'pregame_team_AdjO_rolling_5' in df.columns and 'pregame_team_AdjO_rolling_10' in df.columns:
        # Volatility as absolute difference between windows
        df['derived_adjO_volatility'] = abs(df['pregame_team_AdjO_rolling_5'] - df['pregame_team_AdjO_rolling_10'])
    
    if 'pregame_team_score_rolling_5' in df.columns and 'pregame_team_score_rolling_10' in df.columns:
        df['derived_scoring_volatility'] = abs(df['pregame_team_score_rolling_5'] - df['pregame_team_score_rolling_10'])
    
    # =========================================================================
    # 52. MIN/MAX FEATURES
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjO' in df.columns:
        df['derived_max_adjO'] = df[['pregame_team_season_adjO', 'pregame_opp_season_adjO']].max(axis=1)
        df['derived_min_adjO'] = df[['pregame_team_season_adjO', 'pregame_opp_season_adjO']].min(axis=1)
        df['derived_adjO_range'] = df['derived_max_adjO'] - df['derived_min_adjO']
    
    if 'pregame_team_season_adjD' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_max_adjD'] = df[['pregame_team_season_adjD', 'pregame_opp_season_adjD']].max(axis=1)
        df['derived_min_adjD'] = df[['pregame_team_season_adjD', 'pregame_opp_season_adjD']].min(axis=1)
    
    # =========================================================================
    # 53. GAME QUALITY INDICATORS
    # =========================================================================
    
    if 'pregame_team_season_barthag' in df.columns and 'pregame_opp_season_barthag' in df.columns:
        df['derived_combined_barthag'] = df['pregame_team_season_barthag'] + df['pregame_opp_season_barthag']
        df['derived_min_barthag'] = df[['pregame_team_season_barthag', 'pregame_opp_season_barthag']].min(axis=1)
    
    # =========================================================================
    # 54. HOME/AWAY SPECIFIC AVERAGES (if available)
    # =========================================================================
    
    # These would need home/away splits in the data
    
    # =========================================================================
    # 55. PERCENTILE RANKS (within dataset)
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns:
        df['derived_adjO_percentile'] = df['pregame_team_season_adjO'].rank(pct=True)
    
    if 'pregame_opp_season_adjD' in df.columns:
        df['derived_opp_adjD_percentile'] = df['pregame_opp_season_adjD'].rank(pct=True)
    
    if 'derived_expected_points' in df.columns:
        df['derived_expected_pts_percentile'] = df['derived_expected_points'].rank(pct=True)
    
    # =========================================================================
    # 56. Z-SCORES
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns:
        mean_adjO = df['pregame_team_season_adjO'].mean()
        std_adjO = df['pregame_team_season_adjO'].std()
        if std_adjO > 0:
            df['derived_adjO_zscore'] = (df['pregame_team_season_adjO'] - mean_adjO) / std_adjO
    
    if 'derived_expected_points' in df.columns:
        mean_exp = df['derived_expected_points'].mean()
        std_exp = df['derived_expected_points'].std()
        if std_exp > 0:
            df['derived_expected_pts_zscore'] = (df['derived_expected_points'] - mean_exp) / std_exp
    
    # =========================================================================
    # 57. BINNED/CATEGORICAL FEATURES
    # =========================================================================
    
    if 'pregame_total_line' in df.columns:
        # Bin total into categories
        df['derived_total_bin_low'] = (df['pregame_total_line'] < 130).astype(int)
        df['derived_total_bin_med'] = ((df['pregame_total_line'] >= 130) & (df['pregame_total_line'] < 150)).astype(int)
        df['derived_total_bin_high'] = (df['pregame_total_line'] >= 150).astype(int)
    
    if 'pregame_spread_line' in df.columns:
        # Spread bins
        df['derived_spread_close'] = (abs(df['pregame_spread_line']) < 5).astype(int)
        df['derived_spread_medium'] = ((abs(df['pregame_spread_line']) >= 5) & (abs(df['pregame_spread_line']) < 12)).astype(int)
        df['derived_spread_blowout'] = (abs(df['pregame_spread_line']) >= 12).astype(int)
    
    # =========================================================================
    # 58. MULTIPLICATIVE EFFICIENCY SCORE
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        # Geometric mean style
        df['derived_matchup_geometric'] = np.sqrt(df['pregame_team_season_adjO'] * df['pregame_opp_season_adjD'])
    
    # =========================================================================
    # 59. RELATIVE STRENGTH
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjO' in df.columns:
        # Team offense relative to opponent offense
        df['derived_relative_offense'] = safe_div(df['pregame_team_season_adjO'], df['pregame_opp_season_adjO'])
    
    if 'pregame_team_season_adjD' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        # Team defense relative to opponent defense  
        df['derived_relative_defense'] = safe_div(df['pregame_team_season_adjD'], df['pregame_opp_season_adjD'])
    
    # =========================================================================
    # 60. SIMPLE SUMS
    # =========================================================================
    
    if 'pregame_team_season_adjO' in df.columns and 'pregame_opp_season_adjO' in df.columns:
        df['derived_combined_adjO'] = df['pregame_team_season_adjO'] + df['pregame_opp_season_adjO']
    
    if 'pregame_team_season_adjD' in df.columns and 'pregame_opp_season_adjD' in df.columns:
        df['derived_combined_adjD'] = df['pregame_team_season_adjD'] + df['pregame_opp_season_adjD']
    
    # Count derived features created
    derived_cols = [c for c in df.columns if c.startswith('derived_')]
    print(f"  Created {len(derived_cols)} derived features")
    
    return df


def join_data(bart_games, bart_season, espn_team, player_agg):
    """Join all data sources into unified dataset."""
    print("Joining data sources...")
    
    espn_team = calculate_espn_rolling_stats(espn_team)
    espn_team = calculate_espn_team_derived(espn_team)
    player_agg = calculate_player_rolling_stats(player_agg, espn_team)
    
    espn_full = espn_team.merge(
        player_agg.drop(columns=['game_date'], errors='ignore'),
        on=['game_id', 'team'],
        how='left'
    )
    
    print(f"  ESPN + Player aggregates: {len(espn_full)} rows")
    
    espn_full['game_date'] = pd.to_datetime(espn_full['game_date']).dt.date
    bart_games['date'] = pd.to_datetime(bart_games['date']).dt.date
    
    season_cols = [c for c in bart_season.columns if c != 'team_code']
    
    joined_rows = []
    
    for _, espn_row in espn_full.iterrows():
        game_date = espn_row['game_date']
        team = espn_row['team']
        opponent = espn_row['opponent']
        home_away = espn_row['home_away']
        
        if home_away == 'home':
            bart_match = bart_games[
                (bart_games['date'] == game_date) &
                (bart_games['home_team'] == team) &
                (bart_games['away_team'] == opponent)
            ]
            prefix_team = 'home'
            prefix_opp = 'away'
        else:
            bart_match = bart_games[
                (bart_games['date'] == game_date) &
                (bart_games['away_team'] == team) &
                (bart_games['home_team'] == opponent)
            ]
            prefix_team = 'away'
            prefix_opp = 'home'
        
        if len(bart_match) == 0:
            continue
        
        bart_row = bart_match.iloc[0]
        
        row = {}
        
        # TARGET
        row['target_score'] = espn_row['score']
        row['target_opponent_score'] = espn_row['opponent_score']
        row['target_spread'] = espn_row['score'] - espn_row['opponent_score']
        row['target_total'] = espn_row['score'] + espn_row['opponent_score']
        row['target_win'] = 1 if espn_row['win'] else 0
        
        # CONTEXT
        row['game_id'] = espn_row['game_id']
        row['game_date'] = game_date
        row['team'] = team
        row['opponent'] = opponent
        row['is_home'] = 1 if home_away == 'home' else 0
        
        # PRE-GAME: Vegas lines
        row['pregame_spread_line'] = espn_row.get('spread', np.nan)
        row['pregame_total_line'] = espn_row.get('total_line', np.nan)
        
        # PRE-GAME: Barttorvik rolling stats for TEAM
        bart_rolling_cols = [c for c in bart_games.columns if 'rolling' in c and c.startswith(f'{prefix_team}_')]
        for col in bart_rolling_cols:
            new_col = 'pregame_team_' + col.replace(f'{prefix_team}_', '')
            row[new_col] = bart_row[col]
        
        # PRE-GAME: Barttorvik rolling stats for OPPONENT
        bart_rolling_cols_opp = [c for c in bart_games.columns if 'rolling' in c and c.startswith(f'{prefix_opp}_')]
        for col in bart_rolling_cols_opp:
            new_col = 'pregame_opp_' + col.replace(f'{prefix_opp}_', '')
            row[new_col] = bart_row[col]
        
        # PRE-GAME: Barttorvik context stats
        row['pregame_team_days_rest'] = bart_row.get(f'{prefix_team}_days_rest', np.nan)
        row['pregame_opp_days_rest'] = bart_row.get(f'{prefix_opp}_days_rest', np.nan)
        row['pregame_team_sos'] = bart_row.get(f'{prefix_team}_sos', np.nan)
        row['pregame_opp_sos'] = bart_row.get(f'{prefix_opp}_sos', np.nan)
        row['pregame_home_advantage'] = bart_row.get('home_advantage', 3.0)
        
        # PRE-GAME: ESPN rolling stats for TEAM
        espn_rolling_cols = [c for c in espn_full.columns if 'rolling' in c]
        for col in espn_rolling_cols:
            row[f'pregame_team_{col}'] = espn_row.get(col, np.nan)
        
        # PRE-GAME: Season stats for TEAM
        team_season = bart_season[bart_season['team_code'] == team]
        if len(team_season) > 0:
            team_season_row = team_season.iloc[0]
            for col in season_cols:
                row[f'pregame_team_{col}'] = team_season_row[col]
        
        # PRE-GAME: Season stats for OPPONENT
        opp_season = bart_season[bart_season['team_code'] == opponent]
        if len(opp_season) > 0:
            opp_season_row = opp_season.iloc[0]
            for col in season_cols:
                row[f'pregame_opp_{col}'] = opp_season_row[col]
        
        # POST-GAME stats
        postgame_cols = [
            'field_goals_made', 'field_goals_attempted',
            'three_pointers_made', 'three_pointers_attempted',
            'free_throws_made', 'free_throws_attempted',
            'offensive_rebounds', 'defensive_rebounds',
            'assists', 'steals', 'blocks', 'turnovers', 'fouls',
            'points_off_turnovers', 'fast_break_points', 'points_in_paint',
            'efg_pct', 'true_shooting', 'three_pt_rate', 'ft_rate',
            'two_pt_made', 'two_pt_pct', 'fg_pct_calc', 'three_pt_pct_calc',
            'ast_to_tov', 'stocks',
            'top_player_pts_per_40', 'top_player_usage', 'top_player_ts',
            'top3_avg_pts_per_40', 'top3_avg_usage',
            'starter_avg_pts_per_40', 'starter_avg_ast_to_tov',
            'bench_pts_per_min', 'scoring_concentration', 'minutes_concentration'
        ]
        for col in postgame_cols:
            if col in espn_row.index:
                row[f'postgame_{col}'] = espn_row[col]
        
        joined_rows.append(row)
    
    joined_df = pd.DataFrame(joined_rows)
    
    # Calculate derived features
    joined_df = calculate_derived_features(joined_df)
    
    pregame_cols = [c for c in joined_df.columns if c.startswith('pregame_')]
    derived_cols = [c for c in joined_df.columns if c.startswith('derived_')]
    postgame_cols = [c for c in joined_df.columns if c.startswith('postgame_')]
    target_cols = [c for c in joined_df.columns if c.startswith('target_')]
    
    print(f"  Final joined dataset: {len(joined_df)} rows")
    print(f"  Pre-game columns (valid predictors): {len(pregame_cols)}")
    print(f"  Derived columns (valid predictors): {len(derived_cols)}")
    print(f"  Post-game columns (analysis only): {len(postgame_cols)}")
    print(f"  Target columns: {len(target_cols)}")
    
    return joined_df


def identify_stat_columns(df):
    """Identify which columns are valid pre-game predictors vs post-game analysis."""
    
    all_cols = df.columns.tolist()
    
    pregame_team_stats = [c for c in all_cols if c.startswith('pregame_team_')]
    pregame_opp_stats = [c for c in all_cols if c.startswith('pregame_opp_')]
    derived_stats = [c for c in all_cols if c.startswith('derived_')]
    pregame_context = ['pregame_spread_line', 'pregame_total_line', 'pregame_home_advantage', 'is_home']
    pregame_context = [c for c in pregame_context if c in all_cols]
    
    postgame_stats = [c for c in all_cols if c.startswith('postgame_')]
    targets = [c for c in all_cols if c.startswith('target_')]
    
    print(f"\nIdentified columns:")
    print(f"  Pre-game team stats: {len(pregame_team_stats)}")
    print(f"  Pre-game opponent stats: {len(pregame_opp_stats)}")
    print(f"  Derived features: {len(derived_stats)}")
    print(f"  Pre-game context: {len(pregame_context)}")
    print(f"  Post-game stats (analysis only): {len(postgame_stats)}")
    print(f"  Targets: {len(targets)}")
    
    return pregame_team_stats, pregame_opp_stats, derived_stats, pregame_context, postgame_stats, targets


def calculate_correlation(x, y):
    """Calculate Pearson correlation with p-value, handling NaN."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 30:
        return np.nan, np.nan, len(x_clean)
    
    try:
        r, p = stats.pearsonr(x_clean, y_clean)
        return r, p, len(x_clean)
    except:
        return np.nan, np.nan, len(x_clean)


def run_individual_correlations(df, stat_lists, target_col='target_score'):
    """Run correlations for individual stats against target."""
    print(f"\nRunning individual stat correlations against {target_col}...")
    
    results = []
    target_values = df[target_col].values.astype(float)
    
    for stat_list, stat_type in stat_lists:
        for stat in stat_list:
            if stat not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[stat]):
                continue
            stat_values = df[stat].values.astype(float)
            r, p, n = calculate_correlation(stat_values, target_values)
            if not np.isnan(r):
                results.append({
                    'stat': stat,
                    'type': stat_type,
                    'correlation': r,
                    'p_value': p,
                    'sample_size': n
                })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df['abs_corr'] = results_df['correlation'].abs()
        results_df = results_df.sort_values('abs_corr', ascending=False)
    
    print(f"  Completed {len(results_df)} individual correlations")
    
    return results_df


def run_matchup_correlations(df, target_col='target_score', min_corr=0.10, max_p=0.01):
    """Run specific matchup-based correlations (offense vs corresponding defense)."""
    print("\nRunning matchup-specific correlations...")
    
    results = []
    target_values = df[target_col].values.astype(float)
    
    # Define specific matchup pairs (team offensive stat vs opponent defensive stat)
    matchup_pairs = [
        # Four Factors
        ('pregame_team_season_efg_off', 'pregame_opp_season_efg_def', 'eFG%'),
        ('pregame_team_season_tov_off', 'pregame_opp_season_tov_def', 'TOV%'),
        ('pregame_team_season_oreb_pct', 'pregame_opp_season_dreb_pct', 'Reb%'),
        ('pregame_team_season_ftr_off', 'pregame_opp_season_ftr_def', 'FTR'),
        # Shot types
        ('pregame_team_season_3pt_off', 'pregame_opp_season_3pt_def', '3PT%'),
        ('pregame_team_season_2pt_off', 'pregame_opp_season_2pt_def', '2PT%'),
        ('pregame_team_season_3pt_rate_off', 'pregame_opp_season_3pt_rate_def', '3PT_Rate'),
        # Efficiency
        ('pregame_team_season_adjO', 'pregame_opp_season_adjD', 'Adj_Eff'),
        ('pregame_team_season_ppp_off', 'pregame_opp_season_ppp_def', 'PPP'),
        # Blocks/Assists
        ('pregame_team_season_blk_off', 'pregame_opp_season_blk_def', 'Block%'),
        ('pregame_team_season_ast_off', 'pregame_opp_season_ast_def', 'Assist%'),
        # Rolling
        ('pregame_team_AdjO_rolling_10', 'pregame_opp_AdjD_rolling_10', 'AdjO_rolling'),
        ('pregame_team_eFG_off_rolling_10', 'pregame_opp_eFG_def_rolling_10', 'eFG_rolling'),
    ]
    
    for team_stat, opp_stat, matchup_name in matchup_pairs:
        if team_stat not in df.columns or opp_stat not in df.columns:
            continue
        
        team_values = df[team_stat].values.astype(float)
        opp_values = df[opp_stat].values.astype(float)
        
        # Difference
        diff_values = team_values - opp_values
        r, p, n = calculate_correlation(diff_values, target_values)
        if not np.isnan(r) and abs(r) >= min_corr and p <= max_p:
            results.append({
                'matchup': matchup_name,
                'formula': f'{team_stat} - {opp_stat}',
                'combination': 'difference',
                'correlation': r,
                'p_value': p,
                'sample_size': n
            })
        
        # Ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_values = np.where(opp_values > 0, team_values / opp_values, np.nan)
        r, p, n = calculate_correlation(ratio_values, target_values)
        if not np.isnan(r) and abs(r) >= min_corr and p <= max_p:
            results.append({
                'matchup': matchup_name,
                'formula': f'{team_stat} / {opp_stat}',
                'combination': 'ratio',
                'correlation': r,
                'p_value': p,
                'sample_size': n
            })
        
        # Product
        prod_values = team_values * opp_values
        r, p, n = calculate_correlation(prod_values, target_values)
        if not np.isnan(r) and abs(r) >= min_corr and p <= max_p:
            results.append({
                'matchup': matchup_name,
                'formula': f'{team_stat} * {opp_stat}',
                'combination': 'product',
                'correlation': r,
                'p_value': p,
                'sample_size': n
            })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df['abs_corr'] = results_df['correlation'].abs()
        results_df = results_df.sort_values('abs_corr', ascending=False)
    
    print(f"  Found {len(results_df)} significant matchup correlations")
    
    return results_df


def check_redundancy(df, top_stats, threshold=0.85):
    """Check for redundant stats among top predictors."""
    print(f"\nChecking redundancy among top {len(top_stats)} stats...")
    
    stat_values = {}
    for stat in top_stats:
        if stat in df.columns:
            stat_values[stat] = df[stat].values.astype(float)
    
    redundant_pairs = []
    stats_list = list(stat_values.keys())
    
    for i, stat1 in enumerate(stats_list):
        for stat2 in stats_list[i+1:]:
            r, p, n = calculate_correlation(stat_values[stat1], stat_values[stat2])
            if not np.isnan(r) and abs(r) >= threshold:
                redundant_pairs.append({
                    'stat1': stat1,
                    'stat2': stat2,
                    'correlation': r
                })
    
    return pd.DataFrame(redundant_pairs)


def print_results(individual_results, matchup_results, redundancy_results):
    """Print results to console."""
    
    print("\n" + "="*100)
    print("PHASE 0 RESULTS: CORRELATION ANALYSIS (WITH DERIVED FEATURES)")
    print("="*100)
    
    # Top individual stats
    print("\n" + "-"*100)
    print("TOP 60 INDIVIDUAL STAT CORRELATIONS WITH SCORE")
    print("-"*100)
    print(f"{'Rank':<6} {'Stat':<60} {'Type':<15} {'Corr':>8} {'p-value':>12}")
    print("-"*100)
    
    top_individual = individual_results.head(60)
    for i, (idx, row) in enumerate(top_individual.iterrows()):
        stat_display = row['stat'][:58] if len(row['stat']) > 58 else row['stat']
        print(f"{i+1:<6} {stat_display:<60} {row['type']:<15} {row['correlation']:>8.4f} {row['p_value']:>12.2e}")
    
    # Derived features specifically
    print("\n" + "-"*100)
    print("TOP 30 DERIVED FEATURES")
    print("-"*100)
    derived = individual_results[individual_results['type'] == 'derived'].head(30)
    for i, (idx, row) in enumerate(derived.iterrows()):
        stat_display = row['stat'][:70] if len(row['stat']) > 70 else row['stat']
        print(f"{i+1:<4} {stat_display:<72} r={row['correlation']:>7.4f}")
    
    # Matchup correlations
    if len(matchup_results) > 0:
        print("\n" + "-"*100)
        print("TOP 30 MATCHUP-SPECIFIC CORRELATIONS (Offense vs Corresponding Defense)")
        print("-"*100)
        print(f"{'Matchup':<15} {'Formula':<55} {'Type':<12} {'Corr':>8}")
        print("-"*100)
        
        top_matchup = matchup_results.head(30)
        for i, (idx, row) in enumerate(top_matchup.iterrows()):
            formula_display = row['formula'][:53] if len(row['formula']) > 53 else row['formula']
            print(f"{row['matchup']:<15} {formula_display:<55} {row['combination']:<12} {row['correlation']:>8.4f}")
    
    # Redundancy
    if len(redundancy_results) > 0:
        print("\n" + "-"*100)
        print("REDUNDANT STAT PAIRS (correlation > 0.85)")
        print("-"*100)
        for i, row in redundancy_results.iterrows():
            print(f"  {row['stat1'][:40]} <-> {row['stat2'][:40]}: r={row['correlation']:.3f}")
    
    # Summary
    print("\n" + "="*100)
    print("PHASE 0 SUMMARY")
    print("="*100)
    
    high_priority = individual_results[individual_results['abs_corr'] >= 0.40]
    medium_priority = individual_results[(individual_results['abs_corr'] >= 0.25) & (individual_results['abs_corr'] < 0.40)]
    
    print(f"\nHigh priority stats (|r| >= 0.40): {len(high_priority)}")
    for _, row in high_priority.iterrows():
        print(f"  - {row['stat']} (r={row['correlation']:.3f})")
    
    print(f"\nMedium-High priority stats (0.25 <= |r| < 0.40): {len(medium_priority)}")
    for _, row in medium_priority.head(25).iterrows():
        print(f"  - {row['stat']} (r={row['correlation']:.3f})")
    if len(medium_priority) > 25:
        print(f"  ... and {len(medium_priority) - 25} more")


def build_analysis_dataframe(bart_games, bart_season, espn_team_path, espn_player_path):
    """Build the full analysis dataframe with all features."""
    # Prepare season data
    bart_season = prepare_season_data(bart_season)
    
    # Load ESPN data
    espn_team = pd.read_csv(espn_team_path)
    espn_player = pd.read_csv(espn_player_path)
    
    # Aggregate player stats to team level
    player_agg = aggregate_player_to_team(espn_player, espn_team)
    
    # Join all data
    df = join_data(bart_games, bart_season, espn_team, player_agg)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='NCAAM Phase 0 Correlation Analysis v2')
    parser.add_argument('--bart-games', required=True, help='Path to Barttorvik game data CSV')
    parser.add_argument('--bart-season', required=True, help='Path to Barttorvik season team data CSV')
    parser.add_argument('--espn-team', required=True, help='Path to ESPN team box scores CSV')
    parser.add_argument('--espn-player', required=True, help='Path to ESPN player box scores CSV')
    parser.add_argument('--min-corr', type=float, default=0.10, help='Minimum correlation for matchups (default: 0.10)')
    parser.add_argument('--max-p', type=float, default=0.01, help='Maximum p-value (default: 0.01)')
    
    args = parser.parse_args()
    
    # Load data
    bart_games, bart_season, espn_team, espn_player = load_data(
        args.bart_games, args.bart_season, args.espn_team, args.espn_player
    )
    
    # Prepare season data
    bart_season = prepare_season_data(bart_season)
    
    # Aggregate player stats to team level
    player_agg = aggregate_player_to_team(espn_player, espn_team)
    
    # Join all data
    df = join_data(bart_games, bart_season, espn_team, player_agg)
    
    if len(df) == 0:
        print("ERROR: No data after joining. Check that team codes match between sources.")
        return
    
    # Identify stat columns
    pregame_team, pregame_opp, derived, pregame_context, postgame, targets = identify_stat_columns(df)
    
    # Test against MULTIPLE targets
    target_list = ['target_score', 'target_total', 'target_spread', 'target_win']
    
    for target_col in target_list:
        if target_col not in df.columns:
            continue
            
        print("\n" + "="*100)
        print(f"TESTING AGAINST: {target_col.upper()}")
        print("="*100)
        
        # Run individual correlations
        stat_lists = [
            (pregame_team, 'pregame_team'),
            (pregame_opp, 'pregame_opp'),
            (derived, 'derived'),
            (pregame_context, 'context'),
        ]
        individual_results = run_individual_correlations(df, stat_lists, target_col=target_col)
        
        # Run matchup-specific correlations
        matchup_results = run_matchup_correlations(df, target_col=target_col, min_corr=args.min_corr, max_p=args.max_p)
        
        # Print top results for this target
        print(f"\n--- TOP 40 FEATURES FOR {target_col} ---")
        print(f"{'Rank':<5} {'Stat':<65} {'Corr':>8}")
        print("-"*80)
        for i, (idx, row) in enumerate(individual_results.head(40).iterrows()):
            stat_display = row['stat'][:63] if len(row['stat']) > 63 else row['stat']
            print(f"{i+1:<5} {stat_display:<65} {row['correlation']:>8.4f}")
        
        # Show best matchups
        if len(matchup_results) > 0:
            print(f"\n--- TOP 15 MATCHUP COMBINATIONS FOR {target_col} ---")
            for i, (idx, row) in enumerate(matchup_results.head(15).iterrows()):
                print(f"{i+1:<3} {row['matchup']:<15} {row['combination']:<12} r={row['correlation']:.4f}")
    
    # Check redundancy among top features (using target_score)
    print("\n" + "="*100)
    print("REDUNDANCY CHECK (top 50 features for target_score)")
    print("="*100)
    
    stat_lists = [
        (pregame_team, 'pregame_team'),
        (pregame_opp, 'pregame_opp'),
        (derived, 'derived'),
        (pregame_context, 'context'),
    ]
    score_results = run_individual_correlations(df, stat_lists, target_col='target_score')
    top_stats = score_results.head(50)['stat'].tolist()
    redundancy_results = check_redundancy(df, top_stats)
    
    if len(redundancy_results) > 0:
        print(f"\nFound {len(redundancy_results)} redundant pairs (r > 0.85):")
        for _, row in redundancy_results.head(30).iterrows():
            s1 = row['stat1'][:35] if len(row['stat1']) > 35 else row['stat1']
            s2 = row['stat2'][:35] if len(row['stat2']) > 35 else row['stat2']
            print(f"  {s1:<37} <-> {s2:<37} r={row['correlation']:.3f}")
    
    # Final summary
    print("\n" + "="*100)
    print("FINAL SUMMARY: BEST FEATURES BY TARGET")
    print("="*100)
    
    for target_col in target_list:
        if target_col not in df.columns:
            continue
        stat_lists = [
            (pregame_team, 'pregame_team'),
            (pregame_opp, 'pregame_opp'),
            (derived, 'derived'),
            (pregame_context, 'context'),
        ]
        results = run_individual_correlations(df, stat_lists, target_col=target_col)
        
        print(f"\n{target_col}:")
        for i, (idx, row) in enumerate(results.head(5).iterrows()):
            print(f"  {i+1}. {row['stat']}: r={row['correlation']:.4f}")


def run_extended_exploration(df):
    """Run a ton of exploratory analysis"""
    
    # Fix column references - is_home not pregame_is_home
    if 'is_home' in df.columns and 'pregame_is_home' not in df.columns:
        df['pregame_is_home'] = df['is_home']
    
    # Calculate target_margin if not present
    if 'target_margin' not in df.columns and 'target_score' in df.columns and 'target_opponent_score' in df.columns:
        df['target_margin'] = df['target_score'] - df['target_opponent_score']
    
    results = []
    results.append("=" * 100)
    results.append("EXTENDED EXPLORATION - TESTING ALL KINDS OF SHIT")
    results.append("=" * 100)
    
    # Get all pregame and derived columns
    pregame_cols = [c for c in df.columns if c.startswith('pregame_') or c.startswith('derived_')]
    numeric_cols = [c for c in pregame_cols if df[c].dtype in ['float64', 'int64'] and df[c].notna().sum() > 100]
    
    # ==========================================
    # 1. CORRELATIONS VS TOTAL (both teams combined)
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("1. TOP CORRELATIONS VS GAME TOTAL (predicting combined score)")
    results.append("-" * 100)
    
    if 'target_total' in df.columns:
        total_corrs = []
        for col in numeric_cols:
            valid = df[[col, 'target_total']].dropna()
            if len(valid) > 100:
                r, p = pearsonr(valid[col], valid['target_total'])
                total_corrs.append((col, r, p))
        
        total_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, r, p in total_corrs[:40]:
            results.append(f"  {col:60s} r={r:7.4f}")
    
    # ==========================================
    # 2. CORRELATIONS VS MARGIN (spread prediction)
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("2. TOP CORRELATIONS VS MARGIN (team_score - opp_score)")
    results.append("-" * 100)
    
    if 'target_margin' in df.columns:
        margin_corrs = []
        for col in numeric_cols:
            valid = df[[col, 'target_margin']].dropna()
            if len(valid) > 100:
                r, p = pearsonr(valid[col], valid['target_margin'])
                margin_corrs.append((col, r, p))
        
        margin_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, r, p in margin_corrs[:40]:
            results.append(f"  {col:60s} r={r:7.4f}")
    
    # ==========================================
    # 3. HOME VS AWAY SPLITS
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("3. HOME VS AWAY CORRELATION DIFFERENCES")
    results.append("-" * 100)
    
    if 'pregame_is_home' in df.columns:
        home_df = df[df['pregame_is_home'] == 1]
        away_df = df[df['pregame_is_home'] == 0]
        
        results.append(f"  Home games: {len(home_df)}, Away games: {len(away_df)}")
        results.append(f"\n  Stats where HOME correlation differs significantly from AWAY:")
        
        diff_corrs = []
        for col in numeric_cols[:50]:  # Top 50 to save time
            home_valid = home_df[[col, 'target_score']].dropna()
            away_valid = away_df[[col, 'target_score']].dropna()
            
            if len(home_valid) > 50 and len(away_valid) > 50:
                r_home, _ = pearsonr(home_valid[col], home_valid['target_score'])
                r_away, _ = pearsonr(away_valid[col], away_valid['target_score'])
                diff = r_home - r_away
                diff_corrs.append((col, r_home, r_away, diff))
        
        diff_corrs.sort(key=lambda x: abs(x[3]), reverse=True)
        for col, r_h, r_a, d in diff_corrs[:20]:
            results.append(f"  {col:50s} home={r_h:6.3f} away={r_a:6.3f} diff={d:+6.3f}")
    
    # ==========================================
    # 4. NON-LINEAR RELATIONSHIPS (squared terms)
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("4. NON-LINEAR TESTS (comparing linear vs squared correlation)")
    results.append("-" * 100)
    
    nonlinear_gains = []
    for col in numeric_cols[:40]:
        valid = df[[col, 'target_score']].dropna()
        if len(valid) > 100:
            r_linear, _ = pearsonr(valid[col], valid['target_score'])
            
            # Try squared
            squared = valid[col] ** 2
            r_squared, _ = pearsonr(squared, valid['target_score'])
            
            # Try log (if all positive)
            if (valid[col] > 0).all():
                logged = np.log(valid[col])
                r_log, _ = pearsonr(logged, valid['target_score'])
            else:
                r_log = r_linear
            
            best_transform = 'linear'
            best_r = abs(r_linear)
            if abs(r_squared) > best_r:
                best_transform = 'squared'
                best_r = abs(r_squared)
            if abs(r_log) > best_r:
                best_transform = 'log'
                best_r = abs(r_log)
            
            gain = best_r - abs(r_linear)
            if gain > 0.01:  # Only show if meaningful improvement
                nonlinear_gains.append((col, r_linear, best_transform, best_r, gain))
    
    nonlinear_gains.sort(key=lambda x: x[4], reverse=True)
    results.append("  Stats with >0.01 correlation gain from transformation:")
    for col, r_lin, transform, r_best, gain in nonlinear_gains[:15]:
        results.append(f"  {col:50s} linear={r_lin:6.3f} {transform:8s}={r_best:6.3f} gain={gain:+5.3f}")
    
    if not nonlinear_gains:
        results.append("  No significant non-linear improvements found")
    
    # ==========================================
    # 5. INTERACTION EFFECTS
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("5. INTERACTION EFFECTS (stat × context)")
    results.append("-" * 100)
    
    # Key stats to test interactions with
    key_stats = [
        'pregame_team_season_adjO',
        'pregame_team_AdjO_rolling_10',
        'pregame_team_season_efg_off',
        'pregame_team_season_tempo_raw',
        'derived_expected_points'
    ]
    
    # Context variables to interact with
    context_vars = [
        'pregame_is_home',
        'pregame_days_rest',
        'pregame_opp_season_tempo_raw',
        'pregame_team_sos'
    ]
    
    interaction_corrs = []
    for stat in key_stats:
        if stat not in df.columns:
            continue
        for ctx in context_vars:
            if ctx not in df.columns:
                continue
            
            valid = df[[stat, ctx, 'target_score']].dropna()
            if len(valid) > 100:
                # Base correlation
                r_base, _ = pearsonr(valid[stat], valid['target_score'])
                
                # Interaction
                interaction = valid[stat] * valid[ctx]
                r_interact, _ = pearsonr(interaction, valid['target_score'])
                
                interaction_corrs.append((f"{stat} × {ctx}", r_base, r_interact, r_interact - r_base))
    
    interaction_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    results.append("  Top interaction terms:")
    for name, r_base, r_int, diff in interaction_corrs[:20]:
        results.append(f"  {name:70s} base={r_base:6.3f} interact={r_int:6.3f}")
    
    # ==========================================
    # 6. BINNED ANALYSIS (does relationship change for good vs bad teams?)
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("6. BINNED ANALYSIS - Correlation by team quality tier")
    results.append("-" * 100)
    
    if 'pregame_team_season_adjO' in df.columns:
        # Split into terciles by team quality
        df['_quality_tier'] = pd.qcut(df['pregame_team_season_adjO'], 3, labels=['Bottom', 'Middle', 'Top'])
        
        key_predictors = [
            'derived_expected_points',
            'pregame_total_line',
            'pregame_team_season_tempo_raw',
            'pregame_team_AdjO_rolling_10'
        ]
        
        for pred in key_predictors:
            if pred not in df.columns:
                continue
            results.append(f"\n  {pred}:")
            for tier in ['Bottom', 'Middle', 'Top']:
                tier_df = df[df['_quality_tier'] == tier]
                valid = tier_df[[pred, 'target_score']].dropna()
                if len(valid) > 50:
                    r, _ = pearsonr(valid[pred], valid['target_score'])
                    results.append(f"    {tier:8s} teams (n={len(valid):4d}): r={r:6.3f}")
        
        df.drop('_quality_tier', axis=1, inplace=True)
    
    # ==========================================
    # 7. CONFERENCE GAMES VS NON-CONFERENCE
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("7. CONFERENCE VS NON-CONFERENCE (if data available)")
    results.append("-" * 100)
    
    if 'pregame_conf_game' in df.columns:
        conf_df = df[df['pregame_conf_game'] == 1]
        nonconf_df = df[df['pregame_conf_game'] == 0]
        results.append(f"  Conference games: {len(conf_df)}, Non-conference: {len(nonconf_df)}")
        
        for pred in ['derived_expected_points', 'pregame_total_line', 'pregame_team_season_adjO']:
            if pred not in df.columns:
                continue
            
            conf_valid = conf_df[[pred, 'target_score']].dropna()
            nonconf_valid = nonconf_df[[pred, 'target_score']].dropna()
            
            if len(conf_valid) > 50 and len(nonconf_valid) > 50:
                r_conf, _ = pearsonr(conf_valid[pred], conf_valid['target_score'])
                r_nonconf, _ = pearsonr(nonconf_valid[pred], nonconf_valid['target_score'])
                results.append(f"  {pred:50s} conf={r_conf:6.3f} nonconf={r_nonconf:6.3f}")
    else:
        results.append("  Conference game flag not available")
    
    # ==========================================
    # 8. ROLLING WINDOW COMPARISON
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("8. ROLLING WINDOW COMPARISON (5-game vs 10-game vs season)")
    results.append("-" * 100)
    
    rolling_comparisons = [
        ('AdjO', 'pregame_team_AdjO_rolling_5', 'pregame_team_AdjO_rolling_10', 'pregame_team_season_adjO'),
        ('eFG', 'pregame_team_eFG_off_rolling_5', 'pregame_team_eFG_off_rolling_10', 'pregame_team_season_efg_off'),
        ('Tempo', 'pregame_team_T_rolling_5', 'pregame_team_T_rolling_10', 'pregame_team_season_tempo_raw'),
    ]
    
    for name, r5, r10, season in rolling_comparisons:
        if all(c in df.columns for c in [r5, r10, season]):
            corrs = []
            for col, label in [(r5, '5-game'), (r10, '10-game'), (season, 'season')]:
                valid = df[[col, 'target_score']].dropna()
                if len(valid) > 100:
                    r, _ = pearsonr(valid[col], valid['target_score'])
                    corrs.append(f"{label}={r:.3f}")
            results.append(f"  {name:10s}: {', '.join(corrs)}")
    
    # ==========================================
    # 9. RATIOS AND PERCENTAGES
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("9. ADDITIONAL RATIOS AND COMBINATIONS")
    results.append("-" * 100)
    
    new_features = {}
    
    # Efficiency gap normalized by tempo
    if all(c in df.columns for c in ['derived_net_eff_diff', 'derived_expected_tempo']):
        new_features['eff_gap_per_tempo'] = df['derived_net_eff_diff'] / df['derived_expected_tempo'].replace(0, np.nan)
    
    # Points per possession advantage
    if all(c in df.columns for c in ['pregame_team_season_ppp_off', 'pregame_opp_season_ppp_def']):
        new_features['ppp_advantage'] = df['pregame_team_season_ppp_off'] - df['pregame_opp_season_ppp_def']
    
    # Tempo mismatch × efficiency
    if all(c in df.columns for c in ['derived_pace_mismatch', 'pregame_team_season_adjO']):
        new_features['tempo_mismatch_x_adjO'] = df['derived_pace_mismatch'] * df['pregame_team_season_adjO']
    
    # Home court × efficiency
    if all(c in df.columns for c in ['pregame_is_home', 'pregame_team_season_adjO']):
        new_features['home_x_adjO'] = df['pregame_is_home'] * df['pregame_team_season_adjO']
    
    # Rest × efficiency
    if all(c in df.columns for c in ['pregame_days_rest', 'pregame_team_season_adjO']):
        new_features['rest_x_adjO'] = df['pregame_days_rest'] * df['pregame_team_season_adjO']
    
    # Win pct vs opponent quality
    if all(c in df.columns for c in ['pregame_team_season_wins', 'pregame_opp_season_adjO']):
        new_features['wins_vs_opp_quality'] = df['pregame_team_season_wins'] / df['pregame_opp_season_adjO'].replace(0, np.nan)
    
    # Shooting efficiency combo
    if all(c in df.columns for c in ['pregame_team_season_3pt_off', 'pregame_team_season_2pt_off']):
        new_features['shooting_combo'] = df['pregame_team_season_3pt_off'] * 1.5 + df['pregame_team_season_2pt_off']
    
    # Defensive pressure indicator
    if all(c in df.columns for c in ['pregame_opp_season_tov_def', 'pregame_opp_season_blk_def']):
        new_features['opp_defensive_pressure'] = df['pregame_opp_season_tov_def'] + df['pregame_opp_season_blk_def'] * 2
    
    # Recent form vs season (already have adjO_form, try others)
    if all(c in df.columns for c in ['pregame_team_T_rolling_5', 'pregame_team_season_tempo_raw']):
        new_features['tempo_form'] = df['pregame_team_T_rolling_5'] - df['pregame_team_season_tempo_raw']
    
    # Scoring consistency (if we have game-by-game variance somehow)
    if 'pregame_team_score_rolling_5' in df.columns and 'pregame_team_score_rolling_10' in df.columns:
        new_features['scoring_volatility'] = abs(df['pregame_team_score_rolling_5'] - df['pregame_team_score_rolling_10'])
    
    # Test all new features
    new_corrs = []
    for name, series in new_features.items():
        valid_idx = series.notna() & df['target_score'].notna()
        if valid_idx.sum() > 100:
            r, p = pearsonr(series[valid_idx], df.loc[valid_idx, 'target_score'])
            new_corrs.append((name, r, p))
    
    new_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r, p in new_corrs:
        results.append(f"  {name:50s} r={r:7.4f}")
    
    # ==========================================
    # 10. EXTREME VALUE ANALYSIS
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("10. EXTREME VALUE ANALYSIS (top/bottom 20% of games by score)")
    results.append("-" * 100)
    
    score_20 = df['target_score'].quantile(0.20)
    score_80 = df['target_score'].quantile(0.80)
    
    low_scoring = df[df['target_score'] <= score_20]
    high_scoring = df[df['target_score'] >= score_80]
    
    results.append(f"  Low-scoring games (≤{score_20:.0f} pts): {len(low_scoring)}")
    results.append(f"  High-scoring games (≥{score_80:.0f} pts): {len(high_scoring)}")
    
    results.append(f"\n  Mean predictor values in low vs high scoring games:")
    
    compare_cols = [
        'derived_expected_points',
        'derived_expected_tempo',
        'pregame_team_season_adjO',
        'pregame_team_season_tempo_raw',
        'pregame_opp_season_adjD',
        'pregame_total_line'
    ]
    
    for col in compare_cols:
        if col in df.columns:
            low_mean = low_scoring[col].mean()
            high_mean = high_scoring[col].mean()
            diff_pct = (high_mean - low_mean) / low_mean * 100 if low_mean != 0 else 0
            results.append(f"  {col:50s} low={low_mean:7.2f} high={high_mean:7.2f} diff={diff_pct:+5.1f}%")
    
    # ==========================================
    # 11. UPSET ANALYSIS
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("11. UPSET ANALYSIS (when underdog wins)")
    results.append("-" * 100)
    
    if 'pregame_spread_line' in df.columns and 'target_margin' in df.columns:
        # Underdog = positive spread (team is getting points)
        underdogs = df[df['pregame_spread_line'] > 0].copy()
        underdog_wins = underdogs[underdogs['target_margin'] > 0]  # They won outright
        
        results.append(f"  Underdog games: {len(underdogs)}, Underdog wins: {len(underdog_wins)} ({100*len(underdog_wins)/len(underdogs):.1f}%)")
        
        if len(underdog_wins) > 50:
            results.append(f"\n  What predicts underdog WINS (correlation with margin when underdog):")
            
            upset_corrs = []
            for col in numeric_cols[:30]:
                valid = underdogs[[col, 'target_margin']].dropna()
                if len(valid) > 50:
                    r, p = pearsonr(valid[col], valid['target_margin'])
                    upset_corrs.append((col, r, p))
            
            upset_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            for col, r, p in upset_corrs[:15]:
                results.append(f"    {col:55s} r={r:7.4f}")
    
    # ==========================================
    # 12. PLAYER-LEVEL FEATURES
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("12. PLAYER-LEVEL AGGREGATES")
    results.append("-" * 100)
    
    player_cols = [c for c in df.columns if 'starter' in c.lower() or 'top3' in c.lower() or 'bench' in c.lower()]
    
    if player_cols:
        player_corrs = []
        for col in player_cols:
            valid = df[[col, 'target_score']].dropna()
            if len(valid) > 100:
                r, p = pearsonr(valid[col], valid['target_score'])
                player_corrs.append((col, r))
        
        player_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, r in player_corrs[:20]:
            results.append(f"  {col:60s} r={r:7.4f}")
    else:
        results.append("  No player-level features found")
    
    # ==========================================
    # 13. VEGAS LINE ANALYSIS
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("13. VEGAS LINES - How good are they?")
    results.append("-" * 100)
    
    if 'pregame_total_line' in df.columns and 'target_total' in df.columns:
        valid = df[['pregame_total_line', 'target_total']].dropna()
        r, _ = pearsonr(valid['pregame_total_line'], valid['target_total'])
        mae = (valid['pregame_total_line'] - valid['target_total']).abs().mean()
        rmse = np.sqrt(((valid['pregame_total_line'] - valid['target_total'])**2).mean())
        bias = (valid['pregame_total_line'] - valid['target_total']).mean()
        
        results.append(f"  Vegas Total Line:")
        results.append(f"    Correlation with actual total: r={r:.4f}")
        results.append(f"    MAE: {mae:.2f} points")
        results.append(f"    RMSE: {rmse:.2f} points")
        results.append(f"    Bias: {bias:+.2f} (positive = Vegas overestimates)")
    
    if 'pregame_spread_line' in df.columns and 'target_margin' in df.columns:
        valid = df[['pregame_spread_line', 'target_margin']].dropna()
        # Spread is from team's perspective, so predicted margin = -spread
        predicted_margin = -valid['pregame_spread_line']
        r, _ = pearsonr(predicted_margin, valid['target_margin'])
        mae = (predicted_margin - valid['target_margin']).abs().mean()
        
        results.append(f"\n  Vegas Spread:")
        results.append(f"    Correlation with actual margin: r={r:.4f}")
        results.append(f"    MAE: {mae:.2f} points")
        
        # Cover rate
        covers = ((valid['target_margin'] + valid['pregame_spread_line']) > 0).mean()
        results.append(f"    Team covers spread: {100*covers:.1f}%")
    
    # ==========================================
    # 14. FEATURE IMPORTANCE PREVIEW (simple)
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("14. QUICK FEATURE IMPORTANCE (correlation magnitude ranking)")
    results.append("-" * 100)
    
    # Rank all features by absolute correlation
    all_corrs = []
    for col in numeric_cols:
        valid = df[[col, 'target_score']].dropna()
        if len(valid) > 100:
            r, _ = pearsonr(valid[col], valid['target_score'])
            all_corrs.append((col, abs(r), r))
    
    all_corrs.sort(key=lambda x: x[1], reverse=True)
    
    results.append("  Top 30 features by |correlation| with score:")
    for i, (col, abs_r, r) in enumerate(all_corrs[:30], 1):
        results.append(f"  {i:2d}. {col:55s} |r|={abs_r:.4f} (r={r:+.4f})")
    
    # ==========================================
    # 15. DIMINISHING RETURNS CHECK
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("15. CHECKING FOR DIMINISHING RETURNS")
    results.append("-" * 100)
    results.append("  (Does correlation change when values are extreme?)")
    
    for col in ['derived_expected_points', 'pregame_team_season_adjO', 'pregame_total_line']:
        if col not in df.columns:
            continue
        
        valid = df[[col, 'target_score']].dropna()
        if len(valid) < 200:
            continue
        
        # Split into terciles
        terciles = pd.qcut(valid[col], 3, labels=['Low', 'Med', 'High'])
        
        results.append(f"\n  {col}:")
        for tier in ['Low', 'Med', 'High']:
            tier_data = valid[terciles == tier]
            if len(tier_data) > 50:
                r, _ = pearsonr(tier_data[col], tier_data['target_score'])
                results.append(f"    {tier:5s} tercile (n={len(tier_data):4d}): r={r:.4f}")
    
    # ==========================================
    # 16. TIME-BASED ANALYSIS (early vs late season)
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("16. SEASONAL TIMING ANALYSIS")
    results.append("-" * 100)
    
    if 'date' in df.columns or 'game_date' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'game_date'
        df['_month'] = pd.to_datetime(df[date_col]).dt.month
        
        # Nov-Dec = early, Jan-Feb = mid, Mar = late
        early = df[df['_month'].isin([11, 12])]
        mid = df[df['_month'].isin([1, 2])]
        late = df[df['_month'].isin([3, 4])]
        
        results.append(f"  Early season (Nov-Dec): {len(early)} games")
        results.append(f"  Mid season (Jan-Feb): {len(mid)} games")
        results.append(f"  Late season (Mar-Apr): {len(late)} games")
        
        for pred in ['derived_expected_points', 'pregame_total_line']:
            if pred not in df.columns:
                continue
            results.append(f"\n  {pred}:")
            for period, period_df in [('Early', early), ('Mid', mid), ('Late', late)]:
                valid = period_df[[pred, 'target_score']].dropna()
                if len(valid) > 30:
                    r, _ = pearsonr(valid[pred], valid['target_score'])
                    results.append(f"    {period:6s}: r={r:.4f} (n={len(valid)})")
        
        df.drop('_month', axis=1, inplace=True)
    
    # ==========================================
    # 17. BLOWOUT VS CLOSE GAME ANALYSIS
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("17. BLOWOUT VS CLOSE GAME ANALYSIS")
    results.append("-" * 100)
    
    if 'target_margin' in df.columns:
        abs_margin = df['target_margin'].abs()
        close_games = df[abs_margin <= 10]
        blowouts = df[abs_margin >= 20]
        
        results.append(f"  Close games (margin ≤10): {len(close_games)}")
        results.append(f"  Blowouts (margin ≥20): {len(blowouts)}")
        
        results.append(f"\n  Predictability comparison:")
        for pred in ['derived_expected_points', 'pregame_total_line', 'pregame_team_season_adjO']:
            if pred not in df.columns:
                continue
            
            close_valid = close_games[[pred, 'target_score']].dropna()
            blow_valid = blowouts[[pred, 'target_score']].dropna()
            
            if len(close_valid) > 50 and len(blow_valid) > 50:
                r_close, _ = pearsonr(close_valid[pred], close_valid['target_score'])
                r_blow, _ = pearsonr(blow_valid[pred], blow_valid['target_score'])
                results.append(f"  {pred:50s} close={r_close:.3f} blowout={r_blow:.3f}")
    
    # ==========================================
    # 18. OPPONENT STRENGTH INTERACTION
    # ==========================================
    results.append("\n" + "-" * 100)
    results.append("18. HOW DOES OPPONENT STRENGTH AFFECT PREDICTABILITY?")
    results.append("-" * 100)
    
    if 'pregame_opp_season_adjO' in df.columns:
        df['_opp_tier'] = pd.qcut(df['pregame_opp_season_adjO'], 3, labels=['Weak', 'Medium', 'Strong'])
        
        results.append("  Correlation of expected_points with score BY opponent strength:")
        for tier in ['Weak', 'Medium', 'Strong']:
            tier_df = df[df['_opp_tier'] == tier]
            if 'derived_expected_points' in df.columns:
                valid = tier_df[['derived_expected_points', 'target_score']].dropna()
                if len(valid) > 50:
                    r, _ = pearsonr(valid['derived_expected_points'], valid['target_score'])
                    results.append(f"    vs {tier:7s} opponents (n={len(valid):4d}): r={r:.4f}")
        
        df.drop('_opp_tier', axis=1, inplace=True)
    
    return "\n".join(results)


if __name__ == '__main__':
    # Run main analysis
    main()
    
    # Then run extended exploration
    print("\n\n" + "="*100)
    print("RUNNING EXTENDED EXPLORATION...")
    print("="*100 + "\n")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bart-games', default='../../processed/base_model_game_data_with_rolling.csv')
    parser.add_argument('--bart-season', default='../../processed/ncaam_team_data_final.csv')
    parser.add_argument('--espn-team', default='../../raw/team_stats.csv')
    parser.add_argument('--espn-player', default='../../raw/player_stats.csv')
    args, _ = parser.parse_known_args()
    
    # Load data again for extended exploration
    bart_games = pd.read_csv(args.bart_games)
    bart_season = pd.read_csv(args.bart_season)
    
    # Build the dataframe with all features
    df = build_analysis_dataframe(bart_games, bart_season, args.espn_team, args.espn_player)

    print("****************************************")
    print(list(df.columns))
    print("****************************************")
    
    # Run extended exploration
    extended_results = run_extended_exploration(df)
    print(extended_results)
