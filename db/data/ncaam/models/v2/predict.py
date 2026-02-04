"""
NCAAM Model v2: Prediction Script

Predicts game outcomes using v2 models with derived features.
Falls back to no-Vegas models if Vegas data unavailable.

Usage:
    python ncaam_predict_v2.py --away DUKE --home UNC
    python ncaam_predict_v2.py --away DUKE --home UNC --neutral
    python ncaam_predict_v2.py --away DUKE --home UNC --spread -3.5 --total 145.5

Output format matches v1 for compatibility.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats

# Paths - relative to script location (like v1)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'processed')
TEAM_DATA_PATH = os.path.join(DATA_DIR, 'ncaam_team_data_final.csv')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')

# V2 Model paths (in same directory as script)
MARGIN_MODEL_PATH = os.path.join(SCRIPT_DIR, 'margin_model.json')
TOTAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'total_model.json')
SCORE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'score_model.json')
FEATURES_PATH = os.path.join(SCRIPT_DIR, 'features.json')
METADATA_PATH = os.path.join(SCRIPT_DIR, 'model_metadata.json')

# V2 No-Vegas Model paths (fallback)
MARGIN_MODEL_NO_VEGAS_PATH = os.path.join(SCRIPT_DIR, 'margin_model_no_vegas.json')
TOTAL_MODEL_NO_VEGAS_PATH = os.path.join(SCRIPT_DIR, 'total_model_no_vegas.json')
SCORE_MODEL_NO_VEGAS_PATH = os.path.join(SCRIPT_DIR, 'score_model_no_vegas.json')
FEATURES_NO_VEGAS_PATH = os.path.join(SCRIPT_DIR, 'features_no_vegas.json')
METADATA_NO_VEGAS_PATH = os.path.join(SCRIPT_DIR, 'model_metadata_no_vegas.json')

# Constants
LEAGUE_AVG_TEMPO = 68.0
LEAGUE_AVG_EFFICIENCY = 100.0


def load_models(use_vegas=True):
    """Load v2 models (with or without Vegas features)."""
    
    if use_vegas:
        margin_path = MARGIN_MODEL_PATH
        total_path = TOTAL_MODEL_PATH
        score_path = SCORE_MODEL_PATH
        features_path = FEATURES_PATH
        metadata_path = METADATA_PATH
        model_type = "vegas"
    else:
        margin_path = MARGIN_MODEL_NO_VEGAS_PATH
        total_path = TOTAL_MODEL_NO_VEGAS_PATH
        score_path = SCORE_MODEL_NO_VEGAS_PATH
        features_path = FEATURES_NO_VEGAS_PATH
        metadata_path = METADATA_NO_VEGAS_PATH
        model_type = "no_vegas"
    
    models = {}
    
    # Load margin model
    if os.path.exists(margin_path):
        models['margin'] = xgb.XGBRegressor()
        models['margin'].load_model(margin_path)
    
    # Load total model
    if os.path.exists(total_path):
        models['total'] = xgb.XGBRegressor()
        models['total'].load_model(total_path)
    
    # Load score model
    if os.path.exists(score_path):
        models['score'] = xgb.XGBRegressor()
        models['score'].load_model(score_path)
    
    # Load feature list
    features = []
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
    
    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    metadata['model_type'] = model_type
    
    return models, features, metadata


def load_team_data():
    """Load team season data."""
    df = pd.read_csv(TEAM_DATA_PATH, keep_default_na=False, na_values=[''])
    
    # Map column names
    column_mapping = {
        'Team_Code': 'team_code',
        'Team ID': 'bart_rank',  # Barttorvik overall ranking (1 = best)
        'Adj. Off. Eff': 'adjO',
        'Adj. Def. Eff': 'adjD',
        'Barthag': 'barthag',
        'Eff. FG% Off': 'efg_off',
        'Eff. FG% Def': 'efg_def',
        'FT Rate Off': 'ftr_off',
        'Turnover% Off': 'tov_off',
        'Turnover% Def': 'tov_def',
        'Off. Reb%': 'oreb_pct',
        'Def. Reb%': 'dreb_pct',
        'Raw Tempo': 'tempo_raw',
        '3P% Off': '3pt_off',
        '3P% Def': '3pt_def',
        'Elite SOS': 'elite_sos',
        'Conference': 'conference',
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
    
    # Build lookup dict manually to handle duplicates
    team_lookup = {}
    for _, row in df.iterrows():
        code = row.get('team_code')
        if code and code not in team_lookup:
            team_lookup[code] = row.to_dict()
    
    return team_lookup


def load_game_data():
    """Load game data for rolling stats lookup."""
    return pd.read_csv(GAME_DATA_PATH, na_values=[''], keep_default_na=False)


def get_team_rolling_stats(team_code, game_data):
    """Get the most recent rolling stats for a team."""
    
    def safe_get(series, key, default=None):
        """Safely get value from pandas Series."""
        try:
            val = series[key] if key in series.index else default
            return default if pd.isna(val) else val
        except (KeyError, TypeError):
            return default
    
    # Find games where this team played
    home_games = game_data[game_data['home_team'] == team_code].copy()
    away_games = game_data[game_data['away_team'] == team_code].copy()
    
    if len(home_games) == 0 and len(away_games) == 0:
        return {}
    
    home_games['date'] = pd.to_datetime(home_games['date'])
    away_games['date'] = pd.to_datetime(away_games['date'])
    
    latest_home = home_games.loc[home_games['date'].idxmax()] if len(home_games) > 0 else None
    latest_away = away_games.loc[away_games['date'].idxmax()] if len(away_games) > 0 else None
    
    if latest_home is None and latest_away is None:
        return {}
    elif latest_home is None:
        latest = latest_away
        prefix = 'away'
    elif latest_away is None:
        latest = latest_home
        prefix = 'home'
    else:
        if latest_home['date'] >= latest_away['date']:
            latest = latest_home
            prefix = 'home'
        else:
            latest = latest_away
            prefix = 'away'
    
    # Extract rolling stats using safe_get
    rolling_stats = {
        'AdjO_rolling_5': safe_get(latest, f'{prefix}_AdjO_rolling_5'),
        'AdjO_rolling_10': safe_get(latest, f'{prefix}_AdjO_rolling_10'),
        'AdjD_rolling_5': safe_get(latest, f'{prefix}_AdjD_rolling_5'),
        'AdjD_rolling_10': safe_get(latest, f'{prefix}_AdjD_rolling_10'),
        'T_rolling_5': safe_get(latest, f'{prefix}_T_rolling_5'),
        'T_rolling_10': safe_get(latest, f'{prefix}_T_rolling_10'),
        'days_rest': safe_get(latest, f'{prefix}_days_rest', 3),
        'sos': safe_get(latest, f'{prefix}_sos'),
        'conf': safe_get(latest, f'{prefix}_conf'),
    }
    
    return rolling_stats


def compute_derived_features(team_stats, opp_stats):
    """Compute derived features for prediction."""
    
    derived = {}
    
    # Get values with defaults
    team_adjO = team_stats.get('adjO', LEAGUE_AVG_EFFICIENCY)
    team_adjD = team_stats.get('adjD', LEAGUE_AVG_EFFICIENCY)
    opp_adjO = opp_stats.get('adjO', LEAGUE_AVG_EFFICIENCY)
    opp_adjD = opp_stats.get('adjD', LEAGUE_AVG_EFFICIENCY)
    
    team_tempo = team_stats.get('tempo_raw', LEAGUE_AVG_TEMPO)
    opp_tempo = opp_stats.get('tempo_raw', LEAGUE_AVG_TEMPO)
    
    # Net efficiency difference
    team_net = team_adjO - team_adjD
    opp_net = opp_adjO - opp_adjD
    derived['derived_net_eff_diff'] = team_net - opp_net
    
    # Expected tempo
    derived['derived_expected_tempo'] = (team_tempo + opp_tempo) / 2
    
    # Expected points
    tempo_factor = derived['derived_expected_tempo'] / LEAGUE_AVG_TEMPO
    derived['derived_expected_points'] = (team_adjO + opp_adjD) / 2 * tempo_factor
    
    # Pythagorean
    team_pyth = team_adjO ** 11.5 / (team_adjO ** 11.5 + team_adjD ** 11.5) if team_adjO > 0 and team_adjD > 0 else 0.5
    opp_pyth = opp_adjO ** 11.5 / (opp_adjO ** 11.5 + opp_adjD ** 11.5) if opp_adjO > 0 and opp_adjD > 0 else 0.5
    derived['derived_pyth_diff'] = team_pyth - opp_pyth
    
    # Log5 probability
    if team_pyth > 0 and opp_pyth > 0 and (team_pyth + opp_pyth - 2*team_pyth*opp_pyth) != 0:
        derived['derived_log5_prob'] = (team_pyth * (1 - opp_pyth)) / (team_pyth * (1 - opp_pyth) + opp_pyth * (1 - team_pyth))
    else:
        derived['derived_log5_prob'] = 0.5
    
    # Rank-based features (Barttorvik overall ranking)
    team_rank = team_stats.get('bart_rank')
    opp_rank = opp_stats.get('bart_rank')
    
    if team_rank is not None and opp_rank is not None:
        try:
            team_rank = float(team_rank)
            opp_rank = float(opp_rank)
            
            # Raw rank difference (positive = team ranked worse)
            derived['derived_rank_diff'] = team_rank - opp_rank
            
            # Tier calculation
            def rank_to_tier(rank):
                if rank <= 25:
                    return 1
                elif rank <= 75:
                    return 2
                elif rank <= 150:
                    return 3
                else:
                    return 4
            
            team_tier = rank_to_tier(team_rank)
            opp_tier = rank_to_tier(opp_rank)
            derived['derived_tier_diff'] = opp_tier - team_tier  # positive = team in better tier
            
            # Log rank ratio
            better_rank = min(team_rank, opp_rank)
            worse_rank = max(team_rank, opp_rank)
            if better_rank > 0:
                derived['derived_log_rank_ratio'] = np.log(worse_rank / better_rank)
            else:
                derived['derived_log_rank_ratio'] = 0
            
            # Binary mismatch indicator
            rank_diff_abs = abs(team_rank - opp_rank)
            derived['derived_is_mismatch_100'] = 1 if rank_diff_abs >= 100 else 0
        except (ValueError, TypeError):
            # If rank conversion fails, use defaults
            derived['derived_rank_diff'] = 0
            derived['derived_tier_diff'] = 0
            derived['derived_log_rank_ratio'] = 0
            derived['derived_is_mismatch_100'] = 0
    else:
        # No rank data available
        derived['derived_rank_diff'] = 0
        derived['derived_tier_diff'] = 0
        derived['derived_log_rank_ratio'] = 0
        derived['derived_is_mismatch_100'] = 0
    
    # Quad-based features
    team_weighted_quality = team_stats.get('weighted_quality')
    opp_weighted_quality = opp_stats.get('weighted_quality')
    
    if team_weighted_quality is not None and opp_weighted_quality is not None:
        try:
            derived['derived_weighted_quality_diff'] = float(team_weighted_quality) - float(opp_weighted_quality)
        except (ValueError, TypeError):
            derived['derived_weighted_quality_diff'] = 0
    else:
        derived['derived_weighted_quality_diff'] = 0
    
    # Q1-Q2 win pct diff
    team_q1_q2_pct = team_stats.get('q1_q2_win_pct')
    opp_q1_q2_pct = opp_stats.get('q1_q2_win_pct')
    
    if team_q1_q2_pct is not None and opp_q1_q2_pct is not None:
        try:
            derived['derived_q1_q2_win_pct_diff'] = float(team_q1_q2_pct) - float(opp_q1_q2_pct)
        except (ValueError, TypeError):
            derived['derived_q1_q2_win_pct_diff'] = 0
    else:
        derived['derived_q1_q2_win_pct_diff'] = 0
    
    # Q3-Q4 losses diff
    team_q3_q4_losses = team_stats.get('q3_q4_losses')
    opp_q3_q4_losses = opp_stats.get('q3_q4_losses')
    
    if team_q3_q4_losses is not None and opp_q3_q4_losses is not None:
        try:
            derived['derived_q3_q4_losses_diff'] = float(team_q3_q4_losses) - float(opp_q3_q4_losses)
        except (ValueError, TypeError):
            derived['derived_q3_q4_losses_diff'] = 0
    else:
        derived['derived_q3_q4_losses_diff'] = 0
    
    return derived


def build_features(home_code, away_code, team_data, game_data, spread=None, total_line=None, is_neutral=False):
    """
    Build feature vector for prediction.
    Features are from HOME team perspective (predicting home margin).
    """
    
    # Get team stats
    home_stats = team_data.get(home_code, {})
    away_stats = team_data.get(away_code, {})
    
    # Get rolling stats
    home_rolling = get_team_rolling_stats(home_code, game_data)
    away_rolling = get_team_rolling_stats(away_code, game_data)
    
    # Build feature dict (home = team, away = opp)
    features = {}
    
    # Derived features
    derived = compute_derived_features(home_stats, away_stats)
    features.update(derived)
    
    # Vegas features
    features['pregame_spread_line'] = spread if spread is not None else np.nan
    features['pregame_total_line'] = total_line if total_line is not None else np.nan
    
    # Team season stats
    features['pregame_team_season_adjO'] = home_stats.get('adjO', np.nan)
    features['pregame_team_season_adjD'] = home_stats.get('adjD', np.nan)
    features['pregame_team_season_tempo_raw'] = home_stats.get('tempo_raw', np.nan)
    features['pregame_team_season_efg_off'] = home_stats.get('efg_off', np.nan)
    features['pregame_team_season_tov_off'] = home_stats.get('tov_off', np.nan)
    features['pregame_team_season_oreb_pct'] = home_stats.get('oreb_pct', np.nan)
    features['pregame_team_season_ftr_off'] = home_stats.get('ftr_off', np.nan)
    features['pregame_team_season_3pt_off'] = home_stats.get('3pt_off', np.nan)
    
    # Opponent season stats
    features['pregame_opp_season_adjO'] = away_stats.get('adjO', np.nan)
    features['pregame_opp_season_adjD'] = away_stats.get('adjD', np.nan)
    features['pregame_opp_season_tempo_raw'] = away_stats.get('tempo_raw', np.nan)
    features['pregame_opp_season_efg_def'] = away_stats.get('efg_def', np.nan)
    features['pregame_opp_season_tov_def'] = away_stats.get('tov_def', np.nan)
    features['pregame_opp_season_dreb_pct'] = away_stats.get('dreb_pct', np.nan)
    features['pregame_opp_season_3pt_def'] = away_stats.get('3pt_def', np.nan)
    
    # Rolling stats
    features['pregame_team_AdjO_rolling_10'] = home_rolling.get('AdjO_rolling_10', np.nan)
    features['pregame_team_AdjD_rolling_10'] = home_rolling.get('AdjD_rolling_10', np.nan)
    features['pregame_opp_AdjO_rolling_10'] = away_rolling.get('AdjO_rolling_10', np.nan)
    features['pregame_opp_AdjD_rolling_10'] = away_rolling.get('AdjD_rolling_10', np.nan)
    
    # Context
    features['is_home'] = 0 if is_neutral else 1
    features['pregame_team_days_rest'] = home_rolling.get('days_rest', 3)
    features['pregame_opp_days_rest'] = away_rolling.get('days_rest', 3)
    features['pregame_team_sos'] = home_stats.get('elite_sos', 0)
    
    return features


def spread_to_win_probability(spread, std_dev=11.0):
    """Convert point spread to win probability."""
    home_win_prob = stats.norm.cdf(spread / std_dev) * 100
    return home_win_prob


def predict(away_code, home_code, spread=None, total_line=None, is_neutral=False):
    """
    Predict game outcome.
    
    Args:
        away_code: Away team code (e.g., "DUKE")
        home_code: Home team code (e.g., "UNC")
        spread: Vegas spread (optional, negative = home favored)
        total_line: Vegas total (optional)
        is_neutral: Whether neutral site game
    
    Returns dict matching v1 format.
    """
    
    # Load data
    team_data = load_team_data()
    game_data = load_game_data()
    
    # Validate teams
    if home_code not in team_data:
        raise ValueError(f"Unknown team: {home_code}")
    if away_code not in team_data:
        raise ValueError(f"Unknown team: {away_code}")
    
    # Determine which models to use
    has_vegas = spread is not None and total_line is not None
    
    # Try Vegas models first if we have Vegas data
    if has_vegas:
        models, feature_list, metadata = load_models(use_vegas=True)
        if not models:
            # Fallback to no-vegas
            models, feature_list, metadata = load_models(use_vegas=False)
            has_vegas = False
    else:
        models, feature_list, metadata = load_models(use_vegas=False)
    
    if not models:
        raise RuntimeError("No models found. Run training script first.")
    
    # Build features
    features = build_features(
        home_code, away_code, team_data, game_data,
        spread=spread if has_vegas else None,
        total_line=total_line if has_vegas else None,
        is_neutral=is_neutral
    )
    
    # Build feature vector in correct order
    X = []
    for f in feature_list:
        val = features.get(f, np.nan)
        # Replace NaN with 0 for prediction (not ideal but allows prediction)
        X.append(0.0 if pd.isna(val) else val)
    X = np.array(X).reshape(1, -1)
    
    # Predict margin (from home team perspective)
    if 'margin' in models:
        predicted_margin = float(models['margin'].predict(X)[0])
    else:
        predicted_margin = 0.0
    
    # Predict total
    if 'total' in models:
        # For total model, we might need to remove total_line from features
        # But since we're using the same feature vector, it should work
        predicted_total = float(models['total'].predict(X)[0])
    else:
        predicted_total = 140.0  # Default
    
    # Derive scores
    home_score = (predicted_total + predicted_margin) / 2
    away_score = (predicted_total - predicted_margin) / 2
    
    # Win probability
    home_win_prob = spread_to_win_probability(predicted_margin)
    away_win_prob = 100 - home_win_prob
    
    # Determine favorite (positive margin = home favored)
    if predicted_margin > 0:
        favorite = home_code
        spread_display = predicted_margin
    else:
        favorite = away_code
        spread_display = abs(predicted_margin)
    
    # Build result matching v1 format
    result = {
        'away_team': {
            'code': away_code,
            'predicted_score': round(away_score, 1),
            'win_probability': round(away_win_prob, 1)
        },
        'home_team': {
            'code': home_code,
            'predicted_score': round(home_score, 1),
            'win_probability': round(home_win_prob, 1)
        },
        'spread': round(spread_display, 1),
        'favorite': favorite,
        'total': round(predicted_total, 1),
        'is_neutral': is_neutral,
        'model_info': {
            'version': 'v2',
            'model_type': metadata.get('model_type', 'unknown'),
            'margin_mae': metadata.get('metrics', {}).get('margin', {}).get('mae'),
            'total_mae': metadata.get('metrics', {}).get('total', {}).get('mae'),
        }
    }
    
    # Add betting edge info if Vegas lines provided
    if spread is not None:
        vegas_predicted_margin = -spread  # Vegas says home wins by -spread
        result['betting'] = {
            'vegas_spread': spread,
            'model_spread': round(-predicted_margin, 1),  # Convert to standard format
            'spread_edge': round(predicted_margin - vegas_predicted_margin, 1),
            'spread_pick': home_code if predicted_margin > vegas_predicted_margin else away_code,
        }
    
    if total_line is not None:
        result['betting'] = result.get('betting', {})
        result['betting']['vegas_total'] = total_line
        result['betting']['model_total'] = round(predicted_total, 1)
        result['betting']['total_edge'] = round(predicted_total - total_line, 1)
        result['betting']['total_pick'] = 'OVER' if predicted_total > total_line else 'UNDER'
    
    return result


def main():
    parser = argparse.ArgumentParser(description='NCAAM Model v2 Prediction')
    parser.add_argument('--away', required=True, help='Away team code')
    parser.add_argument('--home', required=True, help='Home team code')
    parser.add_argument('--spread', type=float, default=None, help='Vegas spread (negative = home favored)')
    parser.add_argument('--total', type=float, default=None, help='Vegas total line')
    parser.add_argument('--neutral', action='store_true', help='Neutral site game')
    
    args = parser.parse_args()
    
    try:
        result = predict(
            args.away, 
            args.home, 
            spread=args.spread, 
            total_line=args.total, 
            is_neutral=args.neutral
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': str(e)}, indent=2))
        exit(1)


if __name__ == '__main__':
    main()