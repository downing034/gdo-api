"""
V4 Prediction Script: Winner-Only Model

Predicts game winner only (no scores).
Returns: favorite, confidence (win probability)
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
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_winner_model.json')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model_config.json')


def safe_float(val, default):
    try:
        return float(val) if val and str(val).strip() != '' else default
    except:
        return default


def load_model():
    """Load trained model and config."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    return model, config


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
                'q1_wins': safe_float(row.get('Q1 Wins'), 0),
                'q1_losses': safe_float(row.get('Q1 Losses'), 0),
                'q2_wins': safe_float(row.get('Q2 Wins'), 0),
                'q2_losses': safe_float(row.get('Q2 Losses'), 0),
                'q12_pct': safe_float(row.get('Q1-Q2 Win Pct'), 0.5),
                'elite_sos': safe_float(row.get('Elite SOS'), 0),
            }
    
    return team_lookup


def load_game_data():
    """Load game data for rolling stats."""
    df = pd.read_csv(GAME_DATA_PATH, keep_default_na=False, na_values=[''])
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_team_rolling_stats(team_code, game_data):
    """Get most recent rolling stats for a team."""
    team_home = game_data[game_data['home_team'] == team_code].copy()
    team_away = game_data[game_data['away_team'] == team_code].copy()
    
    # Get most recent game
    home_cols = {c: c.replace('home_', '') for c in team_home.columns if c.startswith('home_')}
    away_cols = {c: c.replace('away_', '') for c in team_away.columns if c.startswith('away_')}
    
    team_home = team_home.rename(columns=home_cols)
    team_away = team_away.rename(columns=away_cols)
    
    all_games = pd.concat([team_home, team_away]).sort_values('date', ascending=False)
    
    if len(all_games) == 0:
        return {}
    
    latest = all_games.iloc[0]
    
    return {
        'AdjO_rolling_5': safe_float(latest.get('AdjO_rolling_5'), 100),
        'AdjD_rolling_5': safe_float(latest.get('AdjD_rolling_5'), 100),
        'AdjO_rolling_10': safe_float(latest.get('AdjO_rolling_10'), 100),
        'AdjD_rolling_10': safe_float(latest.get('AdjD_rolling_10'), 100),
        'g_score_rolling_5': safe_float(latest.get('g_score_rolling_5'), 0),
        'sos': safe_float(latest.get('sos'), 0),
        'conf': latest.get('conf', ''),
    }


def build_features(home_code, home_stats, away_code, away_stats, team_lookup, is_neutral=False):
    """Build features for winner prediction."""
    features = {}
    
    home_team = team_lookup.get(home_code, {})
    away_team = team_lookup.get(away_code, {})
    
    # === CONTEXT ===
    is_conference = home_stats.get('conf') == away_stats.get('conf')
    features['is_neutral'] = 1.0 if is_neutral else 0.0
    features['is_conference'] = 1.0 if is_conference else 0.0
    features['home_court'] = 0.0 if is_neutral else (2.5 if is_conference else 3.5)
    
    # === SEASON-BASED DIFFERENTIALS ===
    features['quality_diff'] = home_team.get('weighted_quality', 0) - away_team.get('weighted_quality', 0)
    features['quality_diff_sqrt'] = np.sign(features['quality_diff']) * np.sqrt(abs(features['quality_diff']))
    
    home_net_eff = home_team.get('adj_off', 100) - home_team.get('adj_def', 100)
    away_net_eff = away_team.get('adj_off', 100) - away_team.get('adj_def', 100)
    features['net_eff_diff'] = home_net_eff - away_net_eff
    
    features['talent_diff'] = home_team.get('talent', 0) - away_team.get('talent', 0)
    features['adj_off_diff'] = home_team.get('adj_off', 100) - away_team.get('adj_off', 100)
    features['adj_def_diff'] = home_team.get('adj_def', 100) - away_team.get('adj_def', 100)
    features['q12_pct_diff'] = home_team.get('q12_pct', 0.5) - away_team.get('q12_pct', 0.5)
    features['q1_wins_diff'] = home_team.get('q1_wins', 0) - away_team.get('q1_wins', 0)
    
    # === DANGER ZONE DETECTION ===
    abs_quality_diff = abs(features['quality_diff'])
    features['is_danger_zone'] = 1.0 if (abs_quality_diff >= 5 and abs_quality_diff <= 20) else 0.0
    features['is_mismatch'] = 1.0 if abs_quality_diff > 30 else 0.0
    
    # === VARIANCE / VOLATILITY FEATURES ===
    # 3P volatility: reliance on 3s × miss rate
    h_vol3 = home_team.get('3p_rate_off', 35) * (1 - home_team.get('3p_off', 33) / 100)
    a_vol3 = away_team.get('3p_rate_off', 35) * (1 - away_team.get('3p_off', 33) / 100)
    features['vol3_sum'] = h_vol3 + a_vol3
    features['vol3_diff'] = h_vol3 - a_vol3
    
    # Turnover pressure
    features['to_pressure_diff'] = (away_team.get('tov_off', 18) * home_team.get('tov_def', 18)) - \
                                   (home_team.get('tov_off', 18) * away_team.get('tov_def', 18))
    
    # Expected tempo and low possession
    features['expected_tempo'] = (home_team.get('tempo', 68) + away_team.get('tempo', 68)) / 2
    features['low_poss_flag'] = 1.0 if features['expected_tempo'] < 65 else 0.0
    features['quality_x_low_poss'] = features['quality_diff'] * features['low_poss_flag']
    
    # Floor vs Ceiling
    features['floor_diff'] = ((100 - home_team.get('tov_off', 18)) * (100 - home_team.get('ftr_def', 30))) - \
                             ((100 - away_team.get('tov_off', 18)) * (100 - away_team.get('ftr_def', 30)))
    features['ceiling_diff'] = (home_team.get('3p_rate_off', 35) * home_team.get('3p_off', 33) + home_team.get('ftr_off', 30)) - \
                               (away_team.get('3p_rate_off', 35) * away_team.get('3p_off', 33) + away_team.get('ftr_off', 30))
    
    # === MATCHUP-BASED FEATURES ===
    features['home_off_vs_away_def'] = home_team.get('adj_off', 100) - away_team.get('adj_def', 100)
    features['away_off_vs_home_def'] = away_team.get('adj_off', 100) - home_team.get('adj_def', 100)
    features['matchup_diff'] = features['home_off_vs_away_def'] - features['away_off_vs_home_def']
    
    # === 3P FEATURES ===
    features['home_3p_matchup'] = home_team.get('3p_off', 33) - away_team.get('3p_def', 34)
    features['away_3p_matchup'] = away_team.get('3p_off', 33) - home_team.get('3p_def', 34)
    features['threep_matchup_diff'] = features['home_3p_matchup'] - features['away_3p_matchup']
    features['threep_off_diff'] = home_team.get('3p_off', 33) - away_team.get('3p_off', 33)
    
    # 3P boosted in danger zone
    features['threep_matchup_danger'] = features['threep_matchup_diff'] * features['is_danger_zone']
    
    # === FOUR FACTORS DIFFERENTIALS ===
    features['efg_off_diff'] = home_team.get('efg_off', 50) - away_team.get('efg_off', 50)
    features['efg_def_diff'] = home_team.get('efg_def', 50) - away_team.get('efg_def', 50)
    features['tov_off_diff'] = home_team.get('tov_off', 18) - away_team.get('tov_off', 18)
    features['tov_def_diff'] = home_team.get('tov_def', 18) - away_team.get('tov_def', 18)
    features['oreb_diff'] = home_team.get('oreb', 30) - away_team.get('oreb', 30)
    features['dreb_diff'] = home_team.get('dreb', 70) - away_team.get('dreb', 70)
    features['ftr_off_diff'] = home_team.get('ftr_off', 30) - away_team.get('ftr_off', 30)
    features['ftr_def_diff'] = home_team.get('ftr_def', 30) - away_team.get('ftr_def', 30)
    
    # === REBOUNDING & BALL MOVEMENT ===
    features['reb_battle'] = (home_team.get('oreb', 30) - (100 - away_team.get('dreb', 70))) - \
                             (away_team.get('oreb', 30) - (100 - home_team.get('dreb', 70)))
    features['assist_diff'] = home_team.get('assist_off', 50) - away_team.get('assist_off', 50)
    features['block_diff'] = home_team.get('block_def', 10) - away_team.get('block_def', 10)
    features['tov_battle'] = (home_team.get('tov_def', 18) - home_team.get('tov_off', 18)) - \
                             (away_team.get('tov_def', 18) - away_team.get('tov_off', 18))
    
    # TOV and OREB boosted in mismatches (matter less in close games)
    features['tov_battle_mismatch'] = features['tov_battle'] * features['is_mismatch']
    features['oreb_diff_mismatch'] = features['oreb_diff'] * features['is_mismatch']
    
    # === ROLLING FORM FEATURES ===
    features['home_eff_margin_5'] = home_stats['AdjO_rolling_5'] - home_stats['AdjD_rolling_5']
    features['away_eff_margin_5'] = away_stats['AdjO_rolling_5'] - away_stats['AdjD_rolling_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    
    home_eff_margin_10 = home_stats['AdjO_rolling_10'] - home_stats['AdjD_rolling_10']
    away_eff_margin_10 = away_stats['AdjO_rolling_10'] - away_stats['AdjD_rolling_10']
    features['home_momentum'] = features['home_eff_margin_5'] - home_eff_margin_10
    features['away_momentum'] = features['away_eff_margin_5'] - away_eff_margin_10
    
    features['g_score_diff_5'] = home_stats['g_score_rolling_5'] - away_stats['g_score_rolling_5']
    features['sos_diff'] = home_stats.get('sos', 0) - away_stats.get('sos', 0)
    features['elite_sos_diff'] = home_team.get('elite_sos', 0) - away_team.get('elite_sos', 0)
    
    # === OTHER DIFFERENTIALS ===
    features['tempo_diff'] = home_team.get('tempo', 68) - away_team.get('tempo', 68)
    features['experience_diff'] = home_team.get('experience', 2) - away_team.get('experience', 2)
    features['ft_pct_diff'] = home_team.get('ft_pct', 70) - away_team.get('ft_pct', 70)
    features['barthag_diff'] = home_team.get('barthag', 0.5) - away_team.get('barthag', 0.5)
    
    # === RANK DIFFERENTIALS ===
    features['off_rank_diff'] = home_team.get('off_rank', 180) - away_team.get('off_rank', 180)
    features['def_rank_diff'] = home_team.get('def_rank', 180) - away_team.get('def_rank', 180)
    features['combined_rank_diff'] = features['off_rank_diff'] + features['def_rank_diff']
    
    # === INTERACTION FEATURES ===
    features['quality_x_home'] = features['quality_diff'] * features['home_court']
    features['matchup_x_home'] = features['matchup_diff'] * features['home_court']
    features['rank_x_home'] = features['combined_rank_diff'] * features['home_court']
    
    return features


def spread_to_win_probability(spread, std_dev=11.0):
    """Convert spread to win probability."""
    return stats.norm.cdf(spread / std_dev) * 100


def apply_post_prediction_adjustments(predicted_spread, features, team_lookup, home_code, away_code):
    """
    Apply post-prediction adjustments based on error analysis findings.
    
    Key findings:
    - Danger zone (quality diff 5-20): 53% of errors, only 70% accuracy
    - Model overconfident in danger zone
    - Low confidence (<55%) predictions are near coin flip
    - In very low confidence, rank-based pick outperforms model
    
    Returns: adjusted_spread, home_win_prob, std_dev_used, adjustment_notes
    """
    quality_diff = features.get('quality_diff', 0)
    abs_quality_diff = abs(quality_diff)
    combined_rank_diff = features.get('combined_rank_diff', 0)
    
    adjustment_notes = []
    adjusted_spread = predicted_spread
    
    # === VERY LOW CONFIDENCE OVERRIDE ===
    # When model spread is tiny (<1.5 points) AND we're in danger zone,
    # the model is basically guessing. Use rank as tiebreaker.
    # Negative combined_rank_diff means home has better (lower) rank
    if abs(predicted_spread) < 1.5 and 5 <= abs_quality_diff <= 20:
        # Model is uncertain - use rank signal
        # If ranks strongly disagree with model, consider flipping
        rank_signal = -combined_rank_diff / 50  # Normalize: negative = home better
        
        # If model says home by tiny margin but ranks say away is much better (or vice versa)
        if predicted_spread > 0 and combined_rank_diff > 75:
            # Model says home, but away has much better rank - flip to away
            adjusted_spread = -1.0
            adjustment_notes.append("rank_override_to_away")
        elif predicted_spread < 0 and combined_rank_diff < -75:
            # Model says away, but home has much better rank - flip to home
            adjusted_spread = 1.0
            adjustment_notes.append("rank_override_to_home")
    
    # === CONFIDENCE CALCULATION ===
    # Use larger std_dev in danger zone to reduce overconfidence
    if 5 <= abs_quality_diff <= 20:
        std_dev = 13.0
        adjustment_notes.append("danger_zone_penalty")
    elif abs(adjusted_spread) < 2.0:
        std_dev = 14.0
        adjustment_notes.append("tossup_uncertainty")
    else:
        std_dev = 11.0
    
    # Calculate probability with adjusted std_dev
    home_win_prob = stats.norm.cdf(adjusted_spread / std_dev) * 100
    
    return adjusted_spread, home_win_prob, std_dev, adjustment_notes


def predict(away_code, home_code, is_neutral=False, apply_adjustments=True):
    """
    Predict game winner.
    
    Args:
        away_code: Away team code
        home_code: Home team code  
        is_neutral: True for neutral site games (no home court advantage)
        apply_adjustments: Apply post-prediction calibration adjustments (default True)
    """
    # Load model and data
    model, config = load_model()
    game_data = load_game_data()
    team_lookup = load_team_data()
    
    # Get rolling stats
    away_stats = get_team_rolling_stats(away_code, game_data)
    home_stats = get_team_rolling_stats(home_code, game_data)
    
    # Check if teams exist
    if home_code not in team_lookup:
        raise ValueError(f"Unknown home team: {home_code}")
    if away_code not in team_lookup:
        raise ValueError(f"Unknown away team: {away_code}")
    
    # Build features
    features = build_features(home_code, home_stats, away_code, away_stats, team_lookup, is_neutral)
    
    # Create DataFrame
    feature_list = config['features']
    X = pd.DataFrame([{f: features.get(f, 0) for f in feature_list}])
    
    # Predict spread (positive = home wins)
    predicted_spread = float(model.predict(X)[0])
    raw_spread = predicted_spread  # Keep original for output
    
    # Apply post-prediction adjustments
    if apply_adjustments:
        adjusted_spread, home_win_prob, std_dev_used, adjustment_notes = apply_post_prediction_adjustments(
            predicted_spread, features, team_lookup, home_code, away_code
        )
    else:
        adjusted_spread = predicted_spread
        home_win_prob = spread_to_win_probability(predicted_spread)
        std_dev_used = 11.0
        adjustment_notes = []
    
    # Determine winner based on adjusted spread
    if adjusted_spread > 0:
        favorite = home_code
    else:
        favorite = away_code
    
    away_win_prob = 100 - home_win_prob
    confidence = max(home_win_prob, away_win_prob)
    
    # Determine game type for output
    abs_quality_diff = abs(features.get('quality_diff', 0))
    if abs_quality_diff > 30:
        game_type = "mismatch"
    elif abs_quality_diff >= 5:
        game_type = "competitive"  
    else:
        game_type = "tossup"
    
    return {
        'favorite': favorite,
        'confidence': round(confidence, 1),
        'home_team': {
            'code': home_code,
            'win_probability': round(home_win_prob, 1)
        },
        'away_team': {
            'code': away_code,
            'win_probability': round(away_win_prob, 1)
        },
        'predicted_spread': round(predicted_spread, 1),
        'is_neutral': is_neutral,
        'game_type': game_type,
        'adjustments_applied': adjustment_notes,
        'model_info': {
            'version': 'v4',
            'type': 'winner_only',
            'description': 'Moneyline-optimized winner prediction'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Predict NCAAM winner (V4 Winner-Only)')
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