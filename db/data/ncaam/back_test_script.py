"""
Backtest script that compares models using database predictions.

Pulls actual predictions from database for V1, V2, etc.
Trains and runs a test model for comparison.
Prints side-by-side performance metrics.

Usage:
    python backtest_db.py --start 2026-01-02 --end 2026-01-31

Requirements:
    pip install psycopg2-binary pandas numpy xgboost
"""

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Please install psycopg2: pip install psycopg2-binary")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database connection - update these or use environment variables
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'gdo_development'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

# Paths for training data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'processed')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')
TEAM_DATA_PATH = os.path.join(DATA_DIR, 'ncaam_team_data_final.csv')

# Models to pull from database - easy to add more
DB_MODELS = ['v1', 'v2_vegas', 'v2_no_vegas']


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)


def fetch_predictions_from_db(start_date, end_date, model_versions=None):
    """
    Fetch predictions and results from database.
    
    Returns DataFrame with columns:
        game_id, start_time, home_team_code, away_team_code,
        home_score, away_score, spread_line, total_line,
        {model}_home_pred, {model}_away_pred for each model
    """
    if model_versions is None:
        model_versions = DB_MODELS
    
    conn = get_db_connection()
    
    # Build the query
    # Get games with results in date range for NCAAM league
    query = """
    WITH game_data AS (
        SELECT 
            g.id as game_id,
            g.start_time,
            home_team.code as home_team_code,
            away_team.code as away_team_code,
            gr.home_score,
            gr.away_score
        FROM games g
        JOIN leagues l ON g.league_id = l.id
        JOIN teams home_team ON g.home_team_id = home_team.id
        JOIN teams away_team ON g.away_team_id = away_team.id
        JOIN game_results gr ON g.id = gr.game_id
        WHERE l.code = 'ncaam'
          AND g.start_time >= %s
          AND g.start_time < %s
          AND gr.home_score IS NOT NULL
          AND gr.away_score IS NOT NULL
    ),
    odds_data AS (
        SELECT DISTINCT ON (go.game_id)
            go.game_id,
            CASE 
                WHEN go.spread_favorite_team_id = g.home_team_id THEN -go.spread_value
                ELSE go.spread_value
            END as spread_line,
            go.total_line
        FROM game_odds go
        JOIN games g ON go.game_id = g.id
        WHERE go.spread_value IS NOT NULL
        ORDER BY go.game_id, go.fetched_at DESC
    )
    SELECT 
        gd.*,
        od.spread_line,
        od.total_line
    FROM game_data gd
    LEFT JOIN odds_data od ON gd.game_id = od.game_id
    ORDER BY gd.start_time
    """
    
    df = pd.read_sql(query, conn, params=(start_date, end_date + ' 23:59:59'))
    
    # Now fetch predictions for each model version
    for model in model_versions:
        pred_query = """
        SELECT 
            gp.game_id,
            gp.home_predicted_score,
            gp.away_predicted_score
        FROM game_predictions gp
        JOIN games g ON gp.game_id = g.id
        JOIN leagues l ON g.league_id = l.id
        WHERE l.code = 'ncaam'
          AND gp.model_version = %s
          AND g.start_time >= %s
          AND g.start_time < %s
        """
        
        pred_df = pd.read_sql(pred_query, conn, params=(model, start_date, end_date + ' 23:59:59'))
        
        if len(pred_df) > 0:
            pred_df = pred_df.rename(columns={
                'home_predicted_score': f'{model}_home_pred',
                'away_predicted_score': f'{model}_away_pred'
            })
            df = df.merge(pred_df, on='game_id', how='left')
        else:
            df[f'{model}_home_pred'] = np.nan
            df[f'{model}_away_pred'] = np.nan
    
    conn.close()
    
    return df


# =============================================================================
# TRAINING FUNCTIONS (for test model)
# =============================================================================

def load_team_data():
    """Load team season data with rank/quad info."""
    df = pd.read_csv(TEAM_DATA_PATH, keep_default_na=False, na_values=[''])
    
    team_lookup = {}
    for _, row in df.iterrows():
        code = row.get('Team_Code')
        if code:
            team_lookup[code] = {
                'rank': row.get('Team ID'),
                'weighted_quality': row.get('Weighted Quality'),
            }
    return team_lookup


def load_game_data():
    """Load game data for training."""
    return pd.read_csv(GAME_DATA_PATH, keep_default_na=False, na_values=[''])


def build_features_original(away_stats, home_stats, is_neutral=False):
    """Build features for ORIGINAL V1 model (no rank/quad)."""
    features = {}
    
    # Efficiency margins
    features['home_eff_margin_5'] = home_stats['AdjO_rolling_5'] - home_stats['AdjD_rolling_5']
    features['away_eff_margin_5'] = away_stats['AdjO_rolling_5'] - away_stats['AdjD_rolling_5']
    features['home_eff_margin_10'] = home_stats['AdjO_rolling_10'] - home_stats['AdjD_rolling_10']
    features['away_eff_margin_10'] = away_stats['AdjO_rolling_10'] - away_stats['AdjD_rolling_10']
    
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    features['eff_margin_diff_10'] = features['home_eff_margin_10'] - features['away_eff_margin_10']
    
    # Four Factors - offensive
    features['eFG_off_diff_5'] = home_stats['eFG_off_rolling_5'] - away_stats['eFG_off_rolling_5']
    features['eFG_off_diff_10'] = home_stats['eFG_off_rolling_10'] - away_stats['eFG_off_rolling_10']
    features['TOV_off_diff_5'] = away_stats['TOV_off_rolling_5'] - home_stats['TOV_off_rolling_5']
    features['TOV_off_diff_10'] = away_stats['TOV_off_rolling_10'] - home_stats['TOV_off_rolling_10']
    features['OReb_diff_5'] = home_stats['OReb_rolling_5'] - away_stats['OReb_rolling_5']
    features['OReb_diff_10'] = home_stats['OReb_rolling_10'] - away_stats['OReb_rolling_10']
    features['FTR_off_diff_5'] = home_stats['FTR_off_rolling_5'] - away_stats['FTR_off_rolling_5']
    features['FTR_off_diff_10'] = home_stats['FTR_off_rolling_10'] - away_stats['FTR_off_rolling_10']
    
    # Four Factors - defensive
    features['eFG_def_diff_5'] = away_stats['eFG_def_rolling_5'] - home_stats['eFG_def_rolling_5']
    features['eFG_def_diff_10'] = away_stats['eFG_def_rolling_10'] - home_stats['eFG_def_rolling_10']
    features['TOV_def_diff_5'] = home_stats['TOV_def_rolling_5'] - away_stats['TOV_def_rolling_5']
    features['TOV_def_diff_10'] = home_stats['TOV_def_rolling_10'] - away_stats['TOV_def_rolling_10']
    features['DReb_diff_5'] = home_stats['DReb_rolling_5'] - away_stats['DReb_rolling_5']
    features['DReb_diff_10'] = home_stats['DReb_rolling_10'] - away_stats['DReb_rolling_10']
    features['FTR_def_diff_5'] = away_stats['FTR_def_rolling_5'] - home_stats['FTR_def_rolling_5']
    features['FTR_def_diff_10'] = away_stats['FTR_def_rolling_10'] - home_stats['FTR_def_rolling_10']
    
    # Home court
    features['is_neutral'] = 1 if is_neutral else 0
    is_conference = away_stats.get('conf') == home_stats.get('conf')
    features['is_conference_game'] = 1 if is_conference else 0
    
    if is_neutral:
        features['home_court_advantage'] = 0.0
    elif is_conference:
        features['home_court_advantage'] = 3.0
    else:
        features['home_court_advantage'] = 3.5
    
    # Rest
    rest_diff = (away_stats.get('days_rest') or 3) - (home_stats.get('days_rest') or 3)
    features['rest_diff'] = max(-7, min(7, rest_diff))
    
    # SOS
    features['sos_diff'] = (home_stats.get('sos') or 0) - (away_stats.get('sos') or 0)
    
    # Tempo
    features['home_tempo_5'] = home_stats['T_rolling_5']
    features['away_tempo_5'] = away_stats['T_rolling_5']
    features['avg_tempo_5'] = (home_stats['T_rolling_5'] + away_stats['T_rolling_5']) / 2
    features['tempo_diff_5'] = home_stats['T_rolling_5'] - away_stats['T_rolling_5']
    
    # Game score
    features['g_score_diff_5'] = home_stats['g_score_rolling_5'] - away_stats['g_score_rolling_5']
    features['g_score_diff_10'] = home_stats['g_score_rolling_10'] - away_stats['g_score_rolling_10']
    
    # Combined
    features['combined_AdjO_5'] = home_stats['AdjO_rolling_5'] + away_stats['AdjO_rolling_5']
    features['combined_AdjD_5'] = home_stats['AdjD_rolling_5'] + away_stats['AdjD_rolling_5']
    
    return features


def build_features_enhanced(away_code, away_stats, home_code, home_stats, team_lookup, is_neutral=False):
    """Build features for ENHANCED V1 model (with rank/quad)."""
    features = build_features_original(away_stats, home_stats, is_neutral)
    
    # Add rank and quad features
    home_data = team_lookup.get(home_code, {})
    away_data = team_lookup.get(away_code, {})
    
    home_rank = home_data.get('rank', 180)
    away_rank = away_data.get('rank', 180)
    try:
        home_rank = float(home_rank) if home_rank else 180
        away_rank = float(away_rank) if away_rank else 180
    except (ValueError, TypeError):
        home_rank, away_rank = 180, 180
    features['rank_diff'] = home_rank - away_rank
    
    home_wq = home_data.get('weighted_quality', 0)
    away_wq = away_data.get('weighted_quality', 0)
    try:
        home_wq = float(home_wq) if home_wq else 0
        away_wq = float(away_wq) if away_wq else 0
    except (ValueError, TypeError):
        home_wq, away_wq = 0, 0
    features['weighted_quality_diff'] = home_wq - away_wq
    
    return features


def train_test_model(game_data, team_lookup, train_end_date, enhanced=True):
    """
    Train the test model on data before train_end_date.
    
    Args:
        game_data: DataFrame of historical games
        team_lookup: Dict of team stats
        train_end_date: Don't train on games after this date
        enhanced: If True, include rank/quad features
    
    Returns:
        spread_model, total_model, spread_features, total_features
    """
    model_type = "ENHANCED" if enhanced else "ORIGINAL"
    print(f"Training {model_type} test model on games before {train_end_date}...")
    
    game_data = game_data.copy()
    game_data['date'] = pd.to_datetime(game_data['date'])
    
    # Filter to training data
    train_data = game_data[game_data['date'] < train_end_date].copy()
    
    # Filter early season
    def get_cutoff_date(row):
        season = row['season']
        if pd.isna(season):
            return pd.Timestamp('1900-01-01')
        start_year = int('20' + str(season).split('_')[0])
        return pd.Timestamp(f'{start_year}-11-20')
    
    train_data['cutoff_date'] = train_data.apply(get_cutoff_date, axis=1)
    train_data = train_data[train_data['date'] >= train_data['cutoff_date']]
    
    print(f"  Training games: {len(train_data)}")
    
    # Build features for each game
    rows = []
    for _, game in train_data.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']
        is_neutral = game['venue'] == 'N'
        
        away_stats = {
            'AdjO_rolling_5': game['away_AdjO_rolling_5'],
            'AdjO_rolling_10': game['away_AdjO_rolling_10'],
            'AdjD_rolling_5': game['away_AdjD_rolling_5'],
            'AdjD_rolling_10': game['away_AdjD_rolling_10'],
            'T_rolling_5': game['away_T_rolling_5'],
            'T_rolling_10': game['away_T_rolling_10'],
            'eFG_off_rolling_5': game['away_eFG_off_rolling_5'],
            'eFG_off_rolling_10': game['away_eFG_off_rolling_10'],
            'TOV_off_rolling_5': game['away_TOV_off_rolling_5'],
            'TOV_off_rolling_10': game['away_TOV_off_rolling_10'],
            'OReb_rolling_5': game['away_OReb_rolling_5'],
            'OReb_rolling_10': game['away_OReb_rolling_10'],
            'FTR_off_rolling_5': game['away_FTR_off_rolling_5'],
            'FTR_off_rolling_10': game['away_FTR_off_rolling_10'],
            'eFG_def_rolling_5': game['away_eFG_def_rolling_5'],
            'eFG_def_rolling_10': game['away_eFG_def_rolling_10'],
            'TOV_def_rolling_5': game['away_TOV_def_rolling_5'],
            'TOV_def_rolling_10': game['away_TOV_def_rolling_10'],
            'DReb_rolling_5': game['away_DReb_rolling_5'],
            'DReb_rolling_10': game['away_DReb_rolling_10'],
            'FTR_def_rolling_5': game['away_FTR_def_rolling_5'],
            'FTR_def_rolling_10': game['away_FTR_def_rolling_10'],
            'g_score_rolling_5': game['away_g_score_rolling_5'],
            'g_score_rolling_10': game['away_g_score_rolling_10'],
            'days_rest': game['away_days_rest'] if 'away_days_rest' in game.index and pd.notna(game['away_days_rest']) else 3,
            'sos': game['away_sos'] if 'away_sos' in game.index else None,
            'conf': game['away_conf'] if 'away_conf' in game.index else None,
        }
        
        home_stats = {
            'AdjO_rolling_5': game['home_AdjO_rolling_5'],
            'AdjO_rolling_10': game['home_AdjO_rolling_10'],
            'AdjD_rolling_5': game['home_AdjD_rolling_5'],
            'AdjD_rolling_10': game['home_AdjD_rolling_10'],
            'T_rolling_5': game['home_T_rolling_5'],
            'T_rolling_10': game['home_T_rolling_10'],
            'eFG_off_rolling_5': game['home_eFG_off_rolling_5'],
            'eFG_off_rolling_10': game['home_eFG_off_rolling_10'],
            'TOV_off_rolling_5': game['home_TOV_off_rolling_5'],
            'TOV_off_rolling_10': game['home_TOV_off_rolling_10'],
            'OReb_rolling_5': game['home_OReb_rolling_5'],
            'OReb_rolling_10': game['home_OReb_rolling_10'],
            'FTR_off_rolling_5': game['home_FTR_off_rolling_5'],
            'FTR_off_rolling_10': game['home_FTR_off_rolling_10'],
            'eFG_def_rolling_5': game['home_eFG_def_rolling_5'],
            'eFG_def_rolling_10': game['home_eFG_def_rolling_10'],
            'TOV_def_rolling_5': game['home_TOV_def_rolling_5'],
            'TOV_def_rolling_10': game['home_TOV_def_rolling_10'],
            'DReb_rolling_5': game['home_DReb_rolling_5'],
            'DReb_rolling_10': game['home_DReb_rolling_10'],
            'FTR_def_rolling_5': game['home_FTR_def_rolling_5'],
            'FTR_def_rolling_10': game['home_FTR_def_rolling_10'],
            'g_score_rolling_5': game['home_g_score_rolling_5'],
            'g_score_rolling_10': game['home_g_score_rolling_10'],
            'days_rest': game['home_days_rest'] if 'home_days_rest' in game.index and pd.notna(game['home_days_rest']) else 3,
            'sos': game['home_sos'] if 'home_sos' in game.index else None,
            'conf': game['home_conf'] if 'home_conf' in game.index else None,
        }
        
        # Skip if missing key stats
        if pd.isna(away_stats['AdjO_rolling_5']) or pd.isna(home_stats['AdjO_rolling_5']):
            continue
        
        if enhanced:
            features = build_features_enhanced(away_team, away_stats, home_team, home_stats, team_lookup, is_neutral)
        else:
            features = build_features_original(away_stats, home_stats, is_neutral)
        
        features['spread'] = game['home_score'] - game['away_score']
        features['total'] = game['home_score'] + game['away_score']
        rows.append(features)
    
    df = pd.DataFrame(rows)
    df = df.fillna(df.median())
    
    print(f"  Usable training samples: {len(df)}")
    
    # Feature lists
    spread_features = [
        'eff_margin_diff_5', 'eff_margin_diff_10',
        'eFG_off_diff_5', 'eFG_off_diff_10',
        'TOV_off_diff_5', 'TOV_off_diff_10',
        'OReb_diff_5', 'OReb_diff_10',
        'FTR_off_diff_5', 'FTR_off_diff_10',
        'eFG_def_diff_5', 'eFG_def_diff_10',
        'TOV_def_diff_5', 'TOV_def_diff_10',
        'DReb_diff_5', 'DReb_diff_10',
        'FTR_def_diff_5', 'FTR_def_diff_10',
        'home_court_advantage', 'is_neutral', 'is_conference_game',
        'rest_diff', 'sos_diff',
        'g_score_diff_5', 'g_score_diff_10',
    ]
    
    if enhanced:
        spread_features += ['rank_diff', 'weighted_quality_diff']
    
    total_features = [
        'avg_tempo_5', 'home_tempo_5', 'away_tempo_5',
        'combined_AdjO_5', 'combined_AdjD_5',
        'home_eff_margin_5', 'away_eff_margin_5',
        'eFG_off_diff_5', 'eFG_def_diff_5',
        'TOV_off_diff_5', 'TOV_def_diff_5',
        'tempo_diff_5', 'is_neutral',
    ]
    
    X_spread = df[spread_features]
    y_spread = df['spread']
    
    X_total = df[total_features]
    y_total = df['total']
    
    spread_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0
    )
    spread_model.fit(X_spread, y_spread)
    
    total_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0
    )
    total_model.fit(X_total, y_total)
    
    return spread_model, total_model, spread_features, total_features


def predict_test_model(spread_model, total_model, spread_features, total_features, 
                       game_data, team_lookup, test_df, enhanced=True):
    """
    Run test model predictions on test games.
    
    Returns test_df with added columns: test_home_pred, test_away_pred
    """
    game_data = game_data.copy()
    game_data['date'] = pd.to_datetime(game_data['date'])
    
    test_home_preds = []
    test_away_preds = []
    
    for _, row in test_df.iterrows():
        home_code = row['home_team_code']
        away_code = row['away_team_code']
        game_date = pd.to_datetime(row['start_time'])
        
        # Find the game in our training data to get rolling stats
        # Match by team codes and date
        matching = game_data[
            (game_data['home_team'] == home_code) & 
            (game_data['away_team'] == away_code) &
            (game_data['date'].dt.date == game_date.date())
        ]
        
        if len(matching) == 0:
            # Try reverse (teams might be swapped)
            matching = game_data[
                (game_data['home_team'] == away_code) & 
                (game_data['away_team'] == home_code) &
                (game_data['date'].dt.date == game_date.date())
            ]
            if len(matching) > 0:
                # Swap the prediction
                test_home_preds.append(np.nan)
                test_away_preds.append(np.nan)
                continue
        
        if len(matching) == 0:
            test_home_preds.append(np.nan)
            test_away_preds.append(np.nan)
            continue
        
        game = matching.iloc[0]
        is_neutral = game['venue'] == 'N' if 'venue' in game.index else False
        
        away_stats = {
            'AdjO_rolling_5': game['away_AdjO_rolling_5'],
            'AdjO_rolling_10': game['away_AdjO_rolling_10'],
            'AdjD_rolling_5': game['away_AdjD_rolling_5'],
            'AdjD_rolling_10': game['away_AdjD_rolling_10'],
            'T_rolling_5': game['away_T_rolling_5'],
            'T_rolling_10': game['away_T_rolling_10'],
            'eFG_off_rolling_5': game['away_eFG_off_rolling_5'],
            'eFG_off_rolling_10': game['away_eFG_off_rolling_10'],
            'TOV_off_rolling_5': game['away_TOV_off_rolling_5'],
            'TOV_off_rolling_10': game['away_TOV_off_rolling_10'],
            'OReb_rolling_5': game['away_OReb_rolling_5'],
            'OReb_rolling_10': game['away_OReb_rolling_10'],
            'FTR_off_rolling_5': game['away_FTR_off_rolling_5'],
            'FTR_off_rolling_10': game['away_FTR_off_rolling_10'],
            'eFG_def_rolling_5': game['away_eFG_def_rolling_5'],
            'eFG_def_rolling_10': game['away_eFG_def_rolling_10'],
            'TOV_def_rolling_5': game['away_TOV_def_rolling_5'],
            'TOV_def_rolling_10': game['away_TOV_def_rolling_10'],
            'DReb_rolling_5': game['away_DReb_rolling_5'],
            'DReb_rolling_10': game['away_DReb_rolling_10'],
            'FTR_def_rolling_5': game['away_FTR_def_rolling_5'],
            'FTR_def_rolling_10': game['away_FTR_def_rolling_10'],
            'g_score_rolling_5': game['away_g_score_rolling_5'],
            'g_score_rolling_10': game['away_g_score_rolling_10'],
            'days_rest': game['away_days_rest'] if 'away_days_rest' in game.index and pd.notna(game['away_days_rest']) else 3,
            'sos': game['away_sos'] if 'away_sos' in game.index else None,
            'conf': game['away_conf'] if 'away_conf' in game.index else None,
        }
        
        home_stats = {
            'AdjO_rolling_5': game['home_AdjO_rolling_5'],
            'AdjO_rolling_10': game['home_AdjO_rolling_10'],
            'AdjD_rolling_5': game['home_AdjD_rolling_5'],
            'AdjD_rolling_10': game['home_AdjD_rolling_10'],
            'T_rolling_5': game['home_T_rolling_5'],
            'T_rolling_10': game['home_T_rolling_10'],
            'eFG_off_rolling_5': game['home_eFG_off_rolling_5'],
            'eFG_off_rolling_10': game['home_eFG_off_rolling_10'],
            'TOV_off_rolling_5': game['home_TOV_off_rolling_5'],
            'TOV_off_rolling_10': game['home_TOV_off_rolling_10'],
            'OReb_rolling_5': game['home_OReb_rolling_5'],
            'OReb_rolling_10': game['home_OReb_rolling_10'],
            'FTR_off_rolling_5': game['home_FTR_off_rolling_5'],
            'FTR_off_rolling_10': game['home_FTR_off_rolling_10'],
            'eFG_def_rolling_5': game['home_eFG_def_rolling_5'],
            'eFG_def_rolling_10': game['home_eFG_def_rolling_10'],
            'TOV_def_rolling_5': game['home_TOV_def_rolling_5'],
            'TOV_def_rolling_10': game['home_TOV_def_rolling_10'],
            'DReb_rolling_5': game['home_DReb_rolling_5'],
            'DReb_rolling_10': game['home_DReb_rolling_10'],
            'FTR_def_rolling_5': game['home_FTR_def_rolling_5'],
            'FTR_def_rolling_10': game['home_FTR_def_rolling_10'],
            'g_score_rolling_5': game['home_g_score_rolling_5'],
            'g_score_rolling_10': game['home_g_score_rolling_10'],
            'days_rest': game['home_days_rest'] if 'home_days_rest' in game.index and pd.notna(game['home_days_rest']) else 3,
            'sos': game['home_sos'] if 'home_sos' in game.index else None,
            'conf': game['home_conf'] if 'home_conf' in game.index else None,
        }
        
        # Skip if missing key stats
        if pd.isna(away_stats['AdjO_rolling_5']) or pd.isna(home_stats['AdjO_rolling_5']):
            test_home_preds.append(np.nan)
            test_away_preds.append(np.nan)
            continue
        
        # Build features
        if enhanced:
            features = build_features_enhanced(away_code, away_stats, home_code, home_stats, team_lookup, is_neutral)
        else:
            features = build_features_original(away_stats, home_stats, is_neutral)
        
        # Predict
        X_spread = pd.DataFrame([{f: features[f] for f in spread_features}])
        X_total = pd.DataFrame([{f: features[f] for f in total_features}])
        
        pred_spread = float(spread_model.predict(X_spread)[0])
        pred_total = float(total_model.predict(X_total)[0])
        
        pred_home = (pred_total + pred_spread) / 2
        pred_away = (pred_total - pred_spread) / 2
        
        test_home_preds.append(pred_home)
        test_away_preds.append(pred_away)
    
    test_df = test_df.copy()
    test_df['test_home_pred'] = test_home_preds
    test_df['test_away_pred'] = test_away_preds
    
    return test_df


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(df, model_name, home_col, away_col):
    """
    Calculate performance metrics for a model.
    
    Returns dict with: moneyline, ats, ou, mae, games
    """
    # Filter to games where this model has predictions
    valid = df[df[home_col].notna() & df[away_col].notna()].copy()
    
    if len(valid) == 0:
        return {'games': 0, 'moneyline': None, 'ats': None, 'ou': None, 'mae': None}
    
    # Actual results
    valid['actual_spread'] = valid['home_score'] - valid['away_score']
    valid['actual_total'] = valid['home_score'] + valid['away_score']
    valid['actual_home_win'] = valid['actual_spread'] > 0
    
    # Predicted results
    valid['pred_spread'] = valid[home_col] - valid[away_col]
    valid['pred_total'] = valid[home_col] + valid[away_col]
    valid['pred_home_win'] = valid['pred_spread'] > 0
    
    # MONEYLINE
    ml_correct = (valid['pred_home_win'] == valid['actual_home_win']).sum()
    ml_total = len(valid)
    ml_pct = ml_correct / ml_total if ml_total > 0 else 0
    
    # ATS (against vegas spread)
    # If we have spread_line, calculate ATS
    valid_ats = valid[valid['spread_line'].notna()].copy()
    if len(valid_ats) > 0:
        # spread_line is from home team perspective (negative = home favored)
        # We cover if: actual_spread > spread_line (home beat the spread)
        # We pick home to cover if: pred_spread > spread_line
        valid_ats['pred_home_covers'] = valid_ats['pred_spread'] > valid_ats['spread_line']
        valid_ats['actual_home_covers'] = valid_ats['actual_spread'] > valid_ats['spread_line']
        ats_correct = (valid_ats['pred_home_covers'] == valid_ats['actual_home_covers']).sum()
        ats_total = len(valid_ats)
        ats_pct = ats_correct / ats_total if ats_total > 0 else 0
    else:
        ats_correct, ats_total, ats_pct = 0, 0, 0
    
    # OVER/UNDER (against vegas total)
    valid_ou = valid[valid['total_line'].notna()].copy()
    if len(valid_ou) > 0:
        valid_ou['pred_over'] = valid_ou['pred_total'] > valid_ou['total_line']
        valid_ou['actual_over'] = valid_ou['actual_total'] > valid_ou['total_line']
        ou_correct = (valid_ou['pred_over'] == valid_ou['actual_over']).sum()
        ou_total = len(valid_ou)
        ou_pct = ou_correct / ou_total if ou_total > 0 else 0
    else:
        ou_correct, ou_total, ou_pct = 0, 0, 0
    
    # MAE (score prediction error)
    home_mae = np.abs(valid[home_col] - valid['home_score']).mean()
    away_mae = np.abs(valid[away_col] - valid['away_score']).mean()
    avg_mae = (home_mae + away_mae) / 2
    
    return {
        'games': ml_total,
        'moneyline': f"{ml_correct}/{ml_total} ({ml_pct:.1%})",
        'ml_pct': ml_pct,
        'ats': f"{ats_correct}/{ats_total} ({ats_pct:.1%})" if ats_total > 0 else "N/A",
        'ats_pct': ats_pct,
        'ou': f"{ou_correct}/{ou_total} ({ou_pct:.1%})" if ou_total > 0 else "N/A",
        'ou_pct': ou_pct,
        'mae': f"{avg_mae:.1f}",
        'mae_val': avg_mae,
    }


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def backtest(start_date, end_date, test_model_type='enhanced'):
    """
    Run backtest comparing DB models vs test model.
    
    Args:
        start_date: Start of test period
        end_date: End of test period  
        test_model_type: 'enhanced' (with rank/quad) or 'original' (without)
    """
    print("=" * 70)
    print(f"BACKTEST: {start_date} to {end_date}")
    print("=" * 70)
    
    # Fetch predictions from database
    print("\nFetching predictions from database...")
    try:
        df = fetch_predictions_from_db(start_date, end_date)
        print(f"  Games with results: {len(df)}")
    except Exception as e:
        print(f"  ERROR connecting to database: {e}")
        print("\n  Make sure to set environment variables:")
        print("    export DB_HOST=localhost")
        print("    export DB_NAME=gdo_development")
        print("    export DB_USER=postgres")
        print("    export DB_PASSWORD=yourpassword")
        return
    
    # Check what models we have
    for model in DB_MODELS:
        has_preds = df[f'{model}_home_pred'].notna().sum()
        print(f"  {model}: {has_preds} predictions")
    
    # Load training data and train test model
    print("\nLoading training data...")
    game_data = load_game_data()
    team_lookup = load_team_data()
    print(f"  Historical games: {len(game_data)}")
    print(f"  Teams with stats: {len(team_lookup)}")
    
    # Train test model on data before test period
    enhanced = (test_model_type == 'enhanced')
    spread_model, total_model, spread_features, total_features = train_test_model(
        game_data, team_lookup, pd.Timestamp(start_date), enhanced=enhanced
    )
    
    # Run test model predictions
    print("\nRunning test model predictions...")
    df = predict_test_model(
        spread_model, total_model, spread_features, total_features,
        game_data, team_lookup, df, enhanced=enhanced
    )
    test_preds = df['test_home_pred'].notna().sum()
    print(f"  Test model predictions: {test_preds}")
    
    # Calculate metrics for all models
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    all_metrics = {}
    
    # DB models
    for model in DB_MODELS:
        metrics = calculate_metrics(df, model, f'{model}_home_pred', f'{model}_away_pred')
        all_metrics[model] = metrics
    
    # Test model
    test_name = f"TEST ({test_model_type})"
    metrics = calculate_metrics(df, test_name, 'test_home_pred', 'test_away_pred')
    all_metrics[test_name] = metrics
    
    # Print results
    print(f"\nGames in test period: {len(df)}")
    
    print("\nMONEYLINE")
    print("-" * 50)
    for model, m in all_metrics.items():
        if m['games'] > 0:
            print(f"  {model:<20} {m['moneyline']}")
    
    print("\nATS (Against the Spread)")
    print("-" * 50)
    for model, m in all_metrics.items():
        if m['games'] > 0:
            print(f"  {model:<20} {m['ats']}")
    
    print("\nOVER/UNDER")
    print("-" * 50)
    for model, m in all_metrics.items():
        if m['games'] > 0:
            print(f"  {model:<20} {m['ou']}")
    
    print("\nMAE (Score Prediction Error)")
    print("-" * 50)
    for model, m in all_metrics.items():
        if m['games'] > 0:
            print(f"  {model:<20} {m['mae']}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    # Find best for each metric
    print(f"\n{'Metric':<15} ", end="")
    for model in all_metrics.keys():
        print(f"{model:<18} ", end="")
    print()
    print("-" * 70)
    
    print(f"{'Moneyline':<15} ", end="")
    for model, m in all_metrics.items():
        if m['games'] > 0:
            print(f"{m['ml_pct']:<18.1%} ", end="")
        else:
            print(f"{'N/A':<18} ", end="")
    print()
    
    print(f"{'ATS':<15} ", end="")
    for model, m in all_metrics.items():
        if m['games'] > 0 and m['ats_pct'] > 0:
            print(f"{m['ats_pct']:<18.1%} ", end="")
        else:
            print(f"{'N/A':<18} ", end="")
    print()
    
    print(f"{'O/U':<15} ", end="")
    for model, m in all_metrics.items():
        if m['games'] > 0 and m['ou_pct'] > 0:
            print(f"{m['ou_pct']:<18.1%} ", end="")
        else:
            print(f"{'N/A':<18} ", end="")
    print()
    
    print(f"{'MAE':<15} ", end="")
    for model, m in all_metrics.items():
        if m['games'] > 0:
            print(f"{m['mae_val']:<18.1f} ", end="")
        else:
            print(f"{'N/A':<18} ", end="")
    print()
    
    print("=" * 70)
    
    return df, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest models against database predictions')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--test-model', default='enhanced', choices=['enhanced', 'original'],
                        help='Test model type: enhanced (with rank/quad) or original (without)')
    args = parser.parse_args()
    
    backtest(args.start, args.end, args.test_model)


if __name__ == '__main__':
    main()