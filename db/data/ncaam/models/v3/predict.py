"""
V3 Prediction Script: Dual Score Models

Predicts home_score and away_score separately, then derives:
- Spread = home_score - away_score
- Total = home_score + away_score
"""

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
HOME_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_home_score_model.json')
AWAY_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_away_score_model.json')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model_config.json')


def safe_float(val, default):
    try:
        return float(val) if val and str(val).strip() != '' else default
    except:
        return default


def load_models():
    """Load trained models and config."""
    home_model = xgb.XGBRegressor()
    home_model.load_model(HOME_MODEL_PATH)
    
    away_model = xgb.XGBRegressor()
    away_model.load_model(AWAY_MODEL_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    return home_model, away_model, config


def load_team_data():
    """Load team season data."""
    df = pd.read_csv(TEAM_DATA_PATH, keep_default_na=False, na_values=[''])
    
    team_lookup = {}
    for _, row in df.iterrows():
        code = row.get('Team_Code')
        if code:
            team_lookup[code] = {
                'adj_off': safe_float(row.get('Adj. Off. Eff'), 100),
                'adj_def': safe_float(row.get('Adj. Def. Eff'), 100),
                'off_rank': safe_float(row.get('Adj. Off. Eff Rank'), 180),
                'def_rank': safe_float(row.get('Adj. Def. Eff Rank'), 180),
                'barthag_rank': safe_float(row.get('Team ID'), 180),
                'barthag': safe_float(row.get('Barthag'), 0.5),
                'efg_off': safe_float(row.get('Eff. FG% Off'), 50),
                'efg_def': safe_float(row.get('Eff. FG% Def'), 50),
                'ftr_off': safe_float(row.get('FT Rate Off'), 30),
                'ftr_def': safe_float(row.get('FT Rate Def'), 30),
                'tov_off': safe_float(row.get('Turnover% Off'), 18),
                'tov_def': safe_float(row.get('Turnover% Def'), 18),
                'oreb': safe_float(row.get('Off. Reb%'), 30),
                'dreb': safe_float(row.get('Def. Reb%'), 70),
                '3p_off': safe_float(row.get('3P% Off'), 33),
                '3p_def': safe_float(row.get('3P% Def'), 33),
                '2p_off': safe_float(row.get('2P% Off'), 50),
                '2p_def': safe_float(row.get('2P% Def'), 50),
                'ft_pct': safe_float(row.get('FT% Off'), 70),
                'tempo': safe_float(row.get('Adj. Tempo'), 68),
                'experience': safe_float(row.get('Experience'), 2),
                'talent': safe_float(row.get('Talent'), 0),
                'block_def': safe_float(row.get('Block% Def'), 10),
                'assist_off': safe_float(row.get('Assist% Off'), 50),
                '3p_rate_off': safe_float(row.get('3P Rate Off'), 35),
                'quality_score': safe_float(row.get('Quality Score'), 0),
                'weighted_quality': safe_float(row.get('Weighted Quality'), 0),
            }
    
    return team_lookup


def load_game_data():
    """Load game data for rolling stats."""
    return pd.read_csv(GAME_DATA_PATH, na_values=[''], keep_default_na=False)


def get_team_rolling_stats(team_code, game_data):
    """Get most recent rolling stats for a team."""
    home_games = game_data[game_data['home_team'] == team_code].copy()
    away_games = game_data[game_data['away_team'] == team_code].copy()
    
    home_games['date'] = pd.to_datetime(home_games['date'])
    away_games['date'] = pd.to_datetime(away_games['date'])
    
    latest_home = home_games.loc[home_games['date'].idxmax()] if len(home_games) > 0 else None
    latest_away = away_games.loc[away_games['date'].idxmax()] if len(away_games) > 0 else None
    
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
    
    rolling_stats = {
        'AdjO_rolling_5': latest.get(f'{prefix}_AdjO_rolling_5'),
        'AdjO_rolling_10': latest.get(f'{prefix}_AdjO_rolling_10'),
        'AdjD_rolling_5': latest.get(f'{prefix}_AdjD_rolling_5'),
        'AdjD_rolling_10': latest.get(f'{prefix}_AdjD_rolling_10'),
        'T_rolling_5': latest.get(f'{prefix}_T_rolling_5'),
        'T_rolling_10': latest.get(f'{prefix}_T_rolling_10'),
        'eFG_off_rolling_5': latest.get(f'{prefix}_eFG_off_rolling_5'),
        'eFG_off_rolling_10': latest.get(f'{prefix}_eFG_off_rolling_10'),
        'eFG_def_rolling_5': latest.get(f'{prefix}_eFG_def_rolling_5'),
        'eFG_def_rolling_10': latest.get(f'{prefix}_eFG_def_rolling_10'),
        'TOV_off_rolling_5': latest.get(f'{prefix}_TOV_off_rolling_5'),
        'TOV_off_rolling_10': latest.get(f'{prefix}_TOV_off_rolling_10'),
        'TOV_def_rolling_5': latest.get(f'{prefix}_TOV_def_rolling_5'),
        'TOV_def_rolling_10': latest.get(f'{prefix}_TOV_def_rolling_10'),
        'OReb_rolling_5': latest.get(f'{prefix}_OReb_rolling_5'),
        'OReb_rolling_10': latest.get(f'{prefix}_OReb_rolling_10'),
        'DReb_rolling_5': latest.get(f'{prefix}_DReb_rolling_5'),
        'DReb_rolling_10': latest.get(f'{prefix}_DReb_rolling_10'),
        'FTR_off_rolling_5': latest.get(f'{prefix}_FTR_off_rolling_5'),
        'FTR_off_rolling_10': latest.get(f'{prefix}_FTR_off_rolling_10'),
        'FTR_def_rolling_5': latest.get(f'{prefix}_FTR_def_rolling_5'),
        'FTR_def_rolling_10': latest.get(f'{prefix}_FTR_def_rolling_10'),
        'g_score_rolling_5': latest.get(f'{prefix}_g_score_rolling_5'),
        'g_score_rolling_10': latest.get(f'{prefix}_g_score_rolling_10'),
        'days_rest': latest.get(f'{prefix}_days_rest', 3),
        'sos': latest.get(f'{prefix}_sos'),
        'conf': latest.get(f'{prefix}_conf'),
    }
    
    return rolling_stats


def build_home_features(home_code, home_stats, away_code, away_stats, team_lookup, is_neutral=False):
    """Build features for home score prediction."""
    features = {}
    
    home_team = team_lookup.get(home_code, {})
    away_team = team_lookup.get(away_code, {})
    
    # Quality matchup
    quality_matchup = home_team.get('weighted_quality', 0) - away_team.get('weighted_quality', 0)
    features['quality_matchup'] = quality_matchup
    features['quality_matchup_sqrt'] = np.sign(quality_matchup) * np.sqrt(abs(quality_matchup))
    features['quality_matchup_sq'] = quality_matchup ** 2
    
    # Context
    is_conference = home_stats.get('conf') == away_stats.get('conf')
    features['is_neutral'] = 1.0 if is_neutral else 0.0
    features['is_conference'] = 1.0 if is_conference else 0.0
    features['home_court'] = 0.0 if is_neutral else (2.5 if is_conference else 3.5)
    features['home_days_rest'] = min(max(home_stats.get('days_rest', 3), 0), 14)
    
    # Quality x context
    features['quality_x_home'] = quality_matchup * features['home_court']
    
    # Tempo
    features['avg_tempo_5'] = (home_stats['T_rolling_5'] + away_stats['T_rolling_5']) / 2
    features['avg_tempo_10'] = (home_stats['T_rolling_10'] + away_stats['T_rolling_10']) / 2
    features['avg_tempo_season'] = (home_team.get('tempo', 68) + away_team.get('tempo', 68)) / 2
    
    # Home offensive stats
    features['home_AdjO_10'] = home_stats['AdjO_rolling_10']
    features['home_eFG_off_10'] = home_stats['eFG_off_rolling_10']
    features['home_tempo_10'] = home_stats['T_rolling_10']
    features['home_OReb_5'] = home_stats['OReb_rolling_5']
    features['home_OReb_10'] = home_stats['OReb_rolling_10']
    features['home_FTR_off_10'] = home_stats['FTR_off_rolling_10']
    
    # Home season stats
    features['home_off_rank'] = home_team.get('off_rank', 180)
    features['home_2p_off'] = home_team.get('2p_off', 50)
    features['home_3p_off'] = home_team.get('3p_off', 33)
    features['home_3p_rate_off'] = home_team.get('3p_rate_off', 35)
    features['home_oreb'] = home_team.get('oreb', 30)
    features['home_talent'] = home_team.get('talent', 0)
    features['home_tempo'] = home_team.get('tempo', 68)
    features['home_weighted_quality'] = home_team.get('weighted_quality', 0)
    
    # Away defensive stats
    features['away_AdjD_5'] = away_stats['AdjD_rolling_5']
    features['away_DReb_5'] = away_stats['DReb_rolling_5']
    features['away_FTR_def_10'] = away_stats['FTR_def_rolling_10']
    
    # Away season defensive stats
    features['away_adj_def'] = away_team.get('adj_def', 100)
    features['away_2p_def'] = away_team.get('2p_def', 50)
    features['away_3p_def'] = away_team.get('3p_def', 33)
    features['away_dreb'] = away_team.get('dreb', 70)
    
    # Matchup features
    off_vs_def_eff_10 = home_stats['AdjO_rolling_10'] - away_stats['AdjD_rolling_10']
    off_vs_def_eff_5 = home_stats['AdjO_rolling_5'] - away_stats['AdjD_rolling_5']
    features['off_vs_def_eff_10'] = off_vs_def_eff_10
    features['off_vs_def_eff_5_sqrt'] = np.sign(off_vs_def_eff_5) * np.sqrt(abs(off_vs_def_eff_5))
    
    # Tempo interaction
    features['off_eff_x_tempo'] = home_stats['AdjO_rolling_5'] * features['avg_tempo_5'] / 70
    
    # Context interactions
    features['matchup_x_home'] = off_vs_def_eff_10 * features['home_court']
    features['matchup_x_conf'] = off_vs_def_eff_10 * features['is_conference']
    
    # === NEW: Form/Momentum features (V1-style, helps moneyline) ===
    # Team efficiency margins (who's playing better overall)
    features['home_eff_margin_5'] = home_stats['AdjO_rolling_5'] - home_stats['AdjD_rolling_5']
    features['away_eff_margin_5'] = away_stats['AdjO_rolling_5'] - away_stats['AdjD_rolling_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    
    # Momentum (is team improving or declining?)
    home_eff_margin_10 = home_stats['AdjO_rolling_10'] - home_stats['AdjD_rolling_10']
    away_eff_margin_10 = away_stats['AdjO_rolling_10'] - away_stats['AdjD_rolling_10']
    features['home_momentum'] = features['home_eff_margin_5'] - home_eff_margin_10
    features['away_momentum'] = features['away_eff_margin_5'] - away_eff_margin_10
    
    # G-score diff (composite form metric from V1)
    features['g_score_diff_5'] = home_stats['g_score_rolling_5'] - away_stats['g_score_rolling_5']
    
    # SOS diff (strength of schedule)
    features['sos_diff'] = home_stats.get('sos', 0) - away_stats.get('sos', 0)
    
    # === Defensive Matchup Adjustments ===
    # Away team's 3P defense weakness (positive = bad defense)
    features['opp_3p_def_weakness'] = away_team.get('3p_def', 34) - 34.0
    features['opp_efg_def_weakness'] = away_team.get('efg_def', 51) - 51.0
    
    # Home offense vs away defense matchup
    features['home_3p_vs_def'] = home_team.get('3p_off', 33) - away_team.get('3p_def', 34)
    features['home_efg_vs_def'] = home_team.get('efg_off', 50) - away_team.get('efg_def', 51)
    
    # === Close Game Tiebreaker Features ===
    # FT% advantage
    features['ft_pct_diff'] = home_team.get('ft_pct', 70) - away_team.get('ft_pct', 70)
    
    # Net matchup differential
    features['matchup_diff'] = (home_team.get('adj_off', 100) - away_team.get('adj_def', 100)) - \
                               (away_team.get('adj_off', 100) - home_team.get('adj_def', 100))
    
    # 3P matchup differential
    features['threep_matchup_diff'] = (home_team.get('3p_off', 33) - away_team.get('3p_def', 34)) - \
                                       (away_team.get('3p_off', 33) - home_team.get('3p_def', 34))
    
    # === Rebounding & Ball Movement Features ===
    # Rebounding battle: net second-chance advantage
    features['reb_battle'] = (home_team.get('oreb', 30) - (100 - away_team.get('dreb', 70))) - \
                             (away_team.get('oreb', 30) - (100 - home_team.get('dreb', 70)))
    
    # Assist differential: ball movement / team play
    features['assist_diff'] = home_team.get('assist_off', 50) - away_team.get('assist_off', 50)
    
    # Block differential: rim protection
    features['block_diff'] = home_team.get('block_def', 10) - away_team.get('block_def', 10)
    
    return features


def build_away_features(away_code, away_stats, home_code, home_stats, team_lookup, is_neutral=False):
    """Build features for away score prediction."""
    features = {}
    
    home_team = team_lookup.get(home_code, {})
    away_team = team_lookup.get(away_code, {})
    
    # Quality matchup (away perspective)
    quality_matchup = away_team.get('weighted_quality', 0) - home_team.get('weighted_quality', 0)
    features['quality_matchup_sqrt'] = np.sign(quality_matchup) * np.sqrt(abs(quality_matchup))
    
    # Context
    is_conference = home_stats.get('conf') == away_stats.get('conf')
    features['is_neutral'] = 1.0 if is_neutral else 0.0
    features['is_conference'] = 1.0 if is_conference else 0.0
    features['home_court'] = 0.0 if is_neutral else (-2.5 if is_conference else -3.5)
    features['home_days_rest'] = min(max(home_stats.get('days_rest', 3), 0), 14)
    
    # Tempo
    features['avg_tempo_5'] = (home_stats['T_rolling_5'] + away_stats['T_rolling_5']) / 2
    features['avg_tempo_10'] = (home_stats['T_rolling_10'] + away_stats['T_rolling_10']) / 2
    
    # Away offensive stats
    features['away_AdjO_10'] = away_stats['AdjO_rolling_10']
    features['away_eFG_off_5'] = away_stats['eFG_off_rolling_5']
    features['away_eFG_off_10'] = away_stats['eFG_off_rolling_10']
    features['away_tempo_5'] = away_stats['T_rolling_5']
    features['away_TOV_off_10'] = away_stats['TOV_off_rolling_10']
    features['away_OReb_5'] = away_stats['OReb_rolling_5']
    features['away_OReb_10'] = away_stats['OReb_rolling_10']
    features['away_FTR_off_10'] = away_stats['FTR_off_rolling_10']
    features['away_g_score_10'] = away_stats['g_score_rolling_10']
    
    # Away season stats
    features['away_efg_off'] = away_team.get('efg_off', 50)
    features['away_2p_off'] = away_team.get('2p_off', 50)
    features['away_3p_rate_off'] = away_team.get('3p_rate_off', 35)
    features['away_weighted_quality'] = away_team.get('weighted_quality', 0)
    
    # Home defensive stats
    features['home_AdjD_10'] = home_stats['AdjD_rolling_10']
    features['home_eFG_def_10'] = home_stats['eFG_def_rolling_10']
    features['home_DReb_10'] = home_stats['DReb_rolling_10']
    features['home_FTR_def_10'] = home_stats['FTR_def_rolling_10']
    
    # Home season defensive stats
    features['home_def_rank'] = home_team.get('def_rank', 180)
    features['home_2p_def'] = home_team.get('2p_def', 50)
    features['home_tov_def'] = home_team.get('tov_def', 18)
    
    # Matchup features
    off_vs_def_eff_10 = away_stats['AdjO_rolling_10'] - home_stats['AdjD_rolling_10']
    features['off_vs_def_eff_10'] = off_vs_def_eff_10
    features['off_vs_def_eff_10_sq'] = off_vs_def_eff_10 ** 2
    features['eFG_matchup_5'] = away_stats['eFG_off_rolling_5'] - home_stats['eFG_def_rolling_5']
    features['eFG_matchup_10'] = away_stats['eFG_off_rolling_10'] - home_stats['eFG_def_rolling_10']
    features['FTR_matchup_5'] = away_stats['FTR_off_rolling_5'] - home_stats['FTR_def_rolling_5']
    features['FTR_matchup_10'] = away_stats['FTR_off_rolling_10'] - home_stats['FTR_def_rolling_10']
    
    # Tempo interaction
    features['matchup_x_tempo'] = off_vs_def_eff_10 * features['avg_tempo_10'] / 70
    
    # Context interaction
    features['matchup_x_home'] = off_vs_def_eff_10 * features['home_court']
    
    # === NEW: Form/Momentum features (V1-style, helps moneyline) ===
    # Team efficiency margins (who's playing better overall)
    features['home_eff_margin_5'] = home_stats['AdjO_rolling_5'] - home_stats['AdjD_rolling_5']
    features['away_eff_margin_5'] = away_stats['AdjO_rolling_5'] - away_stats['AdjD_rolling_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    
    # Momentum (is team improving or declining?)
    home_eff_margin_10 = home_stats['AdjO_rolling_10'] - home_stats['AdjD_rolling_10']
    away_eff_margin_10 = away_stats['AdjO_rolling_10'] - away_stats['AdjD_rolling_10']
    features['home_momentum'] = features['home_eff_margin_5'] - home_eff_margin_10
    features['away_momentum'] = features['away_eff_margin_5'] - away_eff_margin_10
    
    # G-score diff (composite form metric from V1)
    features['g_score_diff_5'] = home_stats['g_score_rolling_5'] - away_stats['g_score_rolling_5']
    
    # SOS diff (strength of schedule)
    features['sos_diff'] = home_stats.get('sos', 0) - away_stats.get('sos', 0)
    
    # === Defensive Matchup Adjustments ===
    # Home team's 3P defense weakness (for away scoring)
    features['opp_3p_def_weakness'] = home_team.get('3p_def', 34) - 34.0
    features['opp_efg_def_weakness'] = home_team.get('efg_def', 51) - 51.0
    
    # Away offense vs home defense matchup
    features['away_3p_vs_def'] = away_team.get('3p_off', 33) - home_team.get('3p_def', 34)
    features['away_efg_vs_def'] = away_team.get('efg_off', 50) - home_team.get('efg_def', 51)
    
    # === Close Game Tiebreaker Features ===
    # FT% advantage
    features['ft_pct_diff'] = home_team.get('ft_pct', 70) - away_team.get('ft_pct', 70)
    
    # Net matchup differential
    features['matchup_diff'] = (home_team.get('adj_off', 100) - away_team.get('adj_def', 100)) - \
                               (away_team.get('adj_off', 100) - home_team.get('adj_def', 100))
    
    # 3P matchup differential
    features['threep_matchup_diff'] = (home_team.get('3p_off', 33) - away_team.get('3p_def', 34)) - \
                                       (away_team.get('3p_off', 33) - home_team.get('3p_def', 34))
    
    # === Rebounding & Ball Movement Features ===
    # Rebounding battle: net second-chance advantage
    features['reb_battle'] = (home_team.get('oreb', 30) - (100 - away_team.get('dreb', 70))) - \
                             (away_team.get('oreb', 30) - (100 - home_team.get('dreb', 70)))
    
    # Assist differential: ball movement / team play
    features['assist_diff'] = home_team.get('assist_off', 50) - away_team.get('assist_off', 50)
    
    # Block differential: rim protection
    features['block_diff'] = home_team.get('block_def', 10) - away_team.get('block_def', 10)
    
    return features


def spread_to_win_probability(spread, std_dev=11.0):
    """Convert spread to win probability."""
    return stats.norm.cdf(spread / std_dev) * 100


def predict(away_code, home_code, is_neutral=False):
    """Predict game outcome."""
    # Load models and data
    home_model, away_model, config = load_models()
    game_data = load_game_data()
    team_lookup = load_team_data()
    
    # Get rolling stats
    away_stats = get_team_rolling_stats(away_code, game_data)
    home_stats = get_team_rolling_stats(home_code, game_data)
    
    # Build features
    home_features = build_home_features(home_code, home_stats, away_code, away_stats, team_lookup, is_neutral)
    away_features = build_away_features(away_code, away_stats, home_code, home_stats, team_lookup, is_neutral)
    
    # Create DataFrames
    home_feature_list = config['home_features']
    away_feature_list = config['away_features']
    
    X_home = pd.DataFrame([{f: home_features.get(f, 0) for f in home_feature_list}])
    X_away = pd.DataFrame([{f: away_features.get(f, 0) for f in away_feature_list}])
    
    # Predict individual scores
    predicted_home_score = float(home_model.predict(X_home)[0])
    predicted_away_score = float(away_model.predict(X_away)[0])
    
    # Derive spread and total
    predicted_spread = predicted_home_score - predicted_away_score
    predicted_total = predicted_home_score + predicted_away_score
    
    # Win probability
    home_win_prob = spread_to_win_probability(predicted_spread)
    away_win_prob = 100 - home_win_prob
    
    # Favorite
    if predicted_spread > 0:
        favorite = home_code
        spread_display = predicted_spread
    else:
        favorite = away_code
        spread_display = abs(predicted_spread)
    
    return {
        'away_team': {
            'code': away_code,
            'predicted_score': round(predicted_away_score, 1),
            'win_probability': round(away_win_prob, 1)
        },
        'home_team': {
            'code': home_code,
            'predicted_score': round(predicted_home_score, 1),
            'win_probability': round(home_win_prob, 1)
        },
        'spread': round(spread_display, 1),
        'favorite': favorite,
        'total': round(predicted_total, 1),
        'is_neutral': is_neutral,
        'model_info': {
            'version': 'v3',
            'type': 'dual_score',
            'description': 'Separate home/away score models',
            'home_mae': config.get('home_mae'),
            'away_mae': config.get('away_mae'),
            'spread_mae': config.get('spread_mae')
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Predict NCAAM matchup (V3 Dual Score)')
    parser.add_argument('--away', required=True, help='Away team code')
    parser.add_argument('--home', required=True, help='Home team code')
    parser.add_argument('--neutral', action='store_true', help='Neutral site game')
    args = parser.parse_args()
    
    try:
        result = predict(args.away, args.home, args.neutral)
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(json.dumps({'error': str(e)}))
    except Exception as e:
        print(json.dumps({'error': f'Prediction failed: {str(e)}'}))


if __name__ == '__main__':
    main()