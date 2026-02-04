import os
import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'processed')
TEAM_DATA_PATH = os.path.join(DATA_DIR, 'ncaam_team_data_final.csv')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')
SPREAD_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_spread_model.json')
TOTAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_total_model.json')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model_config.json')


def load_models():
    """Load both trained models."""
    spread_model = xgb.XGBRegressor()
    spread_model.load_model(SPREAD_MODEL_PATH)
    
    total_model = xgb.XGBRegressor()
    total_model.load_model(TOTAL_MODEL_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    return spread_model, total_model, config


def load_team_data():
    """Load team season data."""
    df = pd.read_csv(TEAM_DATA_PATH)
    return df.set_index('Team_Code').to_dict('index')


def load_game_data():
    """Load game data for rolling stats lookup."""
    return pd.read_csv(GAME_DATA_PATH, na_values=[''], keep_default_na=False)


def get_team_rolling_stats(team_code, game_data):
    """
    Get the most recent rolling stats for a team.
    Looks for the team's last appearance in game data.
    """
    # Find games where this team played (as home or away)
    home_games = game_data[game_data['home_team'] == team_code].copy()
    away_games = game_data[game_data['away_team'] == team_code].copy()
    
    # Get the most recent game
    home_games['date'] = pd.to_datetime(home_games['date'])
    away_games['date'] = pd.to_datetime(away_games['date'])
    
    latest_home = home_games.loc[home_games['date'].idxmax()] if len(home_games) > 0 else None
    latest_away = away_games.loc[away_games['date'].idxmax()] if len(away_games) > 0 else None
    
    # Determine which is more recent
    if latest_home is None and latest_away is None:
        raise ValueError(f"No games found for team: {team_code}")
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

    # latest is guaranteed to be set at this point
    assert latest is not None
    
    # Extract rolling stats
    rolling_stats = {
        'AdjO_rolling_5': latest.get(f'{prefix}_AdjO_rolling_5'),
        'AdjO_rolling_10': latest.get(f'{prefix}_AdjO_rolling_10'),
        'AdjD_rolling_5': latest.get(f'{prefix}_AdjD_rolling_5'),
        'AdjD_rolling_10': latest.get(f'{prefix}_AdjD_rolling_10'),
        'T_rolling_5': latest.get(f'{prefix}_T_rolling_5'),
        'T_rolling_10': latest.get(f'{prefix}_T_rolling_10'),
        'eFG_off_rolling_5': latest.get(f'{prefix}_eFG_off_rolling_5'),
        'eFG_off_rolling_10': latest.get(f'{prefix}_eFG_off_rolling_10'),
        'TOV_off_rolling_5': latest.get(f'{prefix}_TOV_off_rolling_5'),
        'TOV_off_rolling_10': latest.get(f'{prefix}_TOV_off_rolling_10'),
        'OReb_rolling_5': latest.get(f'{prefix}_OReb_rolling_5'),
        'OReb_rolling_10': latest.get(f'{prefix}_OReb_rolling_10'),
        'FTR_off_rolling_5': latest.get(f'{prefix}_FTR_off_rolling_5'),
        'FTR_off_rolling_10': latest.get(f'{prefix}_FTR_off_rolling_10'),
        'eFG_def_rolling_5': latest.get(f'{prefix}_eFG_def_rolling_5'),
        'eFG_def_rolling_10': latest.get(f'{prefix}_eFG_def_rolling_10'),
        'TOV_def_rolling_5': latest.get(f'{prefix}_TOV_def_rolling_5'),
        'TOV_def_rolling_10': latest.get(f'{prefix}_TOV_def_rolling_10'),
        'DReb_rolling_5': latest.get(f'{prefix}_DReb_rolling_5'),
        'DReb_rolling_10': latest.get(f'{prefix}_DReb_rolling_10'),
        'FTR_def_rolling_5': latest.get(f'{prefix}_FTR_def_rolling_5'),
        'FTR_def_rolling_10': latest.get(f'{prefix}_FTR_def_rolling_10'),
        'g_score_rolling_5': latest.get(f'{prefix}_g_score_rolling_5'),
        'g_score_rolling_10': latest.get(f'{prefix}_g_score_rolling_10'),
        'days_rest': latest.get(f'{prefix}_days_rest', 3),  # default 3 days
        'sos': latest.get(f'{prefix}_sos'),
        'conf': latest.get(f'{prefix}_conf'),
    }
    
    return rolling_stats


def build_features(away_stats, home_stats, is_neutral=False):
    """
    Build feature dictionaries for spread and total models.
    Mirrors the feature engineering from training.
    """
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
    rest_diff = (home_stats.get('days_rest') or 3) - (away_stats.get('days_rest') or 3)
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
    
    # Combined (for total model)
    features['combined_AdjO_5'] = home_stats['AdjO_rolling_5'] + away_stats['AdjO_rolling_5']
    features['combined_AdjD_5'] = home_stats['AdjD_rolling_5'] + away_stats['AdjD_rolling_5']
    
    return features


def spread_to_win_probability(spread, std_dev=11.0):
    """
    Convert point spread to win probability using normal distribution.
    std_dev of ~11 is typical for college basketball game outcomes.
    """
    # Spread is home - away, so positive spread = home favored
    # Win probability for home team
    home_win_prob = stats.norm.cdf(spread / std_dev) * 100
    return home_win_prob


def predict(away_code, home_code, is_neutral=False):
    """
    Predict game outcome.
    
    Returns spread (positive = home favored), total, and derived scores.
    """
    # Load models and data
    spread_model, total_model, config = load_models()
    game_data = load_game_data()
    
    # Get rolling stats for each team
    away_stats = get_team_rolling_stats(away_code, game_data)
    home_stats = get_team_rolling_stats(home_code, game_data)
    
    # Build features
    features = build_features(away_stats, home_stats, is_neutral)
    
    # Create DataFrames with correct feature order
    spread_features = config['spread_features']
    total_features = config['total_features']
    
    X_spread = pd.DataFrame([{f: features[f] for f in spread_features}])
    X_total = pd.DataFrame([{f: features[f] for f in total_features}])
    
    # Predict
    predicted_spread = float(spread_model.predict(X_spread)[0])
    predicted_total = float(total_model.predict(X_total)[0])
    
    # Derive scores
    home_score = (predicted_total + predicted_spread) / 2
    away_score = (predicted_total - predicted_spread) / 2
    
    # Win probability
    home_win_prob = spread_to_win_probability(predicted_spread)
    away_win_prob = 100 - home_win_prob
    
    # Determine favorite
    if predicted_spread > 0:
        favorite = home_code
        spread_display = predicted_spread
    else:
        favorite = away_code
        spread_display = abs(predicted_spread)
    
    return {
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
            'spread_mae': config.get('spread_mae'),
            'total_mae': config.get('total_mae')
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Predict NCAAM matchup')
    parser.add_argument('--away', required=True, help='Away team code')
    parser.add_argument('--home', required=True, help='Home team code')
    parser.add_argument('--neutral', action='store_true', help='Neutral site game')
    args = parser.parse_args()
    
    result = predict(args.away, args.home, args.neutral)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()