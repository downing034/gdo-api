"""
V4 Training Script: Winner-Only Model (Moneyline Optimized)

Predicts game winner only (no scores).
Uses features optimized for picking winners, including:
- Season-based differentials (stronger signals for winner prediction)
- Rebounding, assist, block differentials
- Quality record differentials (Q1/Q2 wins)

Output: Single classification-style model that predicts spread
        (positive = home wins, negative = away wins)
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'processed')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')
TEAM_DATA_PATH = os.path.join(DATA_DIR, 'ncaam_team_data_final.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_winner_model.json')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model_config.json')


def safe_float(val, default):
    try:
        return float(val) if val and str(val).strip() != '' else default
    except:
        return default


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


def get_cutoff_date(row):
    """Get cutoff date for season filtering."""
    season = row.get('season')
    if season == '2024-25':
        return '2024-12-01'
    elif season == '2023-24':
        return '2023-12-01'
    elif season == '2022-23':
        return '2022-12-01'
    return '2022-12-01'


def load_and_prepare_data(filepath, team_lookup):
    """Load game data and add team season stats."""
    print('Loading game data...')
    df = pd.read_csv(filepath, keep_default_na=False, na_values=[''])
    print(f'  Loaded {len(df)} games')
    
    # Filter by cutoff
    df['date'] = pd.to_datetime(df['date'])
    df['cutoff_date'] = df.apply(get_cutoff_date, axis=1)
    before_filter = len(df)
    df = df[df['date'] >= df['cutoff_date']]
    print(f'  Filtered: {before_filter} → {len(df)}')
    df = df.drop(columns=['cutoff_date'])
    
    # Add team season stats
    season_stats = ['adj_off', 'adj_def', 'off_rank', 'def_rank', 'barthag',
                    'efg_off', 'efg_def', 'ftr_off', 'ftr_def', 'tov_off', 'tov_def',
                    'oreb', 'dreb', '3p_off', '3p_def', '2p_off', '2p_def', 'ft_pct',
                    'tempo', 'experience', 'talent', 'block_def', 'assist_off', '3p_rate_off',
                    'quality_score', 'weighted_quality', 'q1_wins', 'q1_losses', 
                    'q2_wins', 'q2_losses', 'q12_pct', 'elite_sos']
    
    for prefix, col in [('home', 'home_team'), ('away', 'away_team')]:
        for key in season_stats:
            default = 0 if key in ['quality_score', 'weighted_quality', 'talent', 
                                    'q1_wins', 'q1_losses', 'q2_wins', 'q2_losses', 'elite_sos'] else 50
            if key == 'q12_pct':
                default = 0.5
            df[f'{prefix}_{key}'] = df[col].apply(lambda x: team_lookup.get(x, {}).get(key, default))
    
    return df


def engineer_features(df):
    """Engineer features optimized for winner prediction."""
    features = pd.DataFrame(index=df.index)
    
    # === CONTEXT ===
    features['is_neutral'] = (df['venue'] == 'N').astype(float)
    features['is_conference'] = (df['home_conf'] == df['away_conf']).astype(float)
    features['home_court'] = np.where(
        features['is_neutral'] == 1, 0.0,
        np.where(features['is_conference'] == 1, 2.5, 3.5)
    )
    
    # === SEASON-BASED DIFFERENTIALS (strongest signals for winner) ===
    # Quality differential (-5.28 signal - strongest)
    features['quality_diff'] = df['home_weighted_quality'] - df['away_weighted_quality']
    features['quality_diff_sqrt'] = np.sign(features['quality_diff']) * np.sqrt(np.abs(features['quality_diff']))
    
    # Net efficiency differential (-2.49 signal)
    features['home_net_eff'] = df['home_adj_off'] - df['home_adj_def']
    features['away_net_eff'] = df['away_adj_off'] - df['away_adj_def']
    features['net_eff_diff'] = features['home_net_eff'] - features['away_net_eff']
    
    # Talent differential (-1.55 signal)
    features['talent_diff'] = df['home_talent'] - df['away_talent']
    
    # Offensive/Defensive efficiency differentials (-1.34, +1.15 signals)
    features['adj_off_diff'] = df['home_adj_off'] - df['away_adj_off']
    features['adj_def_diff'] = df['home_adj_def'] - df['away_adj_def']  # Lower is better, so positive = home worse
    
    # Quality record differentials (-0.76 signal)
    features['q12_pct_diff'] = df['home_q12_pct'] - df['away_q12_pct']
    features['q1_wins_diff'] = df['home_q1_wins'] - df['away_q1_wins']
    
    # === DANGER ZONE DETECTION ===
    # Quality diff 5-20 is where model struggles (69.6% vs 82.2% elsewhere)
    # In danger zone: 3P diff correlation +0.531, quality diff only +0.166
    abs_quality_diff = np.abs(features['quality_diff'])
    features['is_danger_zone'] = ((abs_quality_diff >= 5) & (abs_quality_diff <= 20)).astype(float)
    features['is_mismatch'] = (abs_quality_diff > 30).astype(float)
    
    # === VARIANCE / VOLATILITY FEATURES ===
    # 3P volatility: reliance on 3s × miss rate (high = more variance)
    features['h_vol3'] = df['home_3p_rate_off'] * (1 - df['home_3p_off'] / 100)
    features['a_vol3'] = df['away_3p_rate_off'] * (1 - df['away_3p_off'] / 100)
    features['vol3_sum'] = features['h_vol3'] + features['a_vol3']
    features['vol3_diff'] = features['h_vol3'] - features['a_vol3']
    
    # Turnover pressure: opponent's sloppiness × team's ability to force TOs
    features['to_pressure_home'] = df['away_tov_off'] * df['home_tov_def']
    features['to_pressure_away'] = df['home_tov_off'] * df['away_tov_def']
    features['to_pressure_diff'] = features['to_pressure_home'] - features['to_pressure_away']
    
    # Expected tempo and low possession flag
    features['expected_tempo'] = (df['home_tempo'] + df['away_tempo']) / 2
    features['low_poss_flag'] = (features['expected_tempo'] < 65).astype(float)
    features['quality_x_low_poss'] = features['quality_diff'] * features['low_poss_flag']
    
    # === FLOOR VS CEILING ===
    # Floor = consistency (low TOs, disciplined D)
    features['h_floor'] = (100 - df['home_tov_off']) * (100 - df['home_ftr_def'])
    features['a_floor'] = (100 - df['away_tov_off']) * (100 - df['away_ftr_def'])
    features['floor_diff'] = features['h_floor'] - features['a_floor']
    
    # Ceiling = upside (3P reliance + FT generation)
    features['h_ceiling'] = df['home_3p_rate_off'] * df['home_3p_off'] + df['home_ftr_off']
    features['a_ceiling'] = df['away_3p_rate_off'] * df['away_3p_off'] + df['away_ftr_off']
    features['ceiling_diff'] = features['h_ceiling'] - features['a_ceiling']
    
    # === MATCHUP-BASED FEATURES ===
    # Offense vs defense matchups
    features['home_off_vs_away_def'] = df['home_adj_off'] - df['away_adj_def']
    features['away_off_vs_home_def'] = df['away_adj_off'] - df['home_adj_def']
    features['matchup_diff'] = features['home_off_vs_away_def'] - features['away_off_vs_home_def']
    
    # === 3P FEATURES (critical in danger zone - 3x more predictive than quality) ===
    features['home_3p_matchup'] = df['home_3p_off'] - df['away_3p_def']
    features['away_3p_matchup'] = df['away_3p_off'] - df['home_3p_def']
    features['threep_matchup_diff'] = features['home_3p_matchup'] - features['away_3p_matchup']
    features['threep_off_diff'] = df['home_3p_off'] - df['away_3p_off']
    
    # 3P boosted in danger zone (where it matters most)
    features['threep_matchup_danger'] = features['threep_matchup_diff'] * features['is_danger_zone']
    
    # === FOUR FACTORS DIFFERENTIALS ===
    features['efg_off_diff'] = df['home_efg_off'] - df['away_efg_off']
    features['efg_def_diff'] = df['home_efg_def'] - df['away_efg_def']
    features['tov_off_diff'] = df['home_tov_off'] - df['away_tov_off']
    features['tov_def_diff'] = df['home_tov_def'] - df['away_tov_def']
    features['oreb_diff'] = df['home_oreb'] - df['away_oreb']
    features['dreb_diff'] = df['home_dreb'] - df['away_dreb']
    features['ftr_off_diff'] = df['home_ftr_off'] - df['away_ftr_off']
    features['ftr_def_diff'] = df['home_ftr_def'] - df['away_ftr_def']
    
    # === REBOUNDING & BALL MOVEMENT (from previous analysis) ===
    # Rebounding battle (-0.98 signal)
    features['reb_battle'] = (df['home_oreb'] - (100 - df['away_dreb'])) - (df['away_oreb'] - (100 - df['home_dreb']))
    
    # Assist differential (-0.74 signal)
    features['assist_diff'] = df['home_assist_off'] - df['away_assist_off']
    
    # Block differential (-0.45 signal)
    features['block_diff'] = df['home_block_def'] - df['away_block_def']
    
    # Turnover battle
    features['tov_battle'] = (df['home_tov_def'] - df['home_tov_off']) - (df['away_tov_def'] - df['away_tov_off'])
    
    # TOV and OREB matter more in mismatches (correlation drops to ~0 in close games)
    features['tov_battle_mismatch'] = features['tov_battle'] * features['is_mismatch']
    features['oreb_diff_mismatch'] = features['oreb_diff'] * features['is_mismatch']
    
    # === ROLLING FORM FEATURES ===
    # Efficiency margins (rolling)
    features['home_eff_margin_5'] = df['home_AdjO_rolling_5'] - df['home_AdjD_rolling_5']
    features['away_eff_margin_5'] = df['away_AdjO_rolling_5'] - df['away_AdjD_rolling_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    
    # Momentum
    home_eff_margin_10 = df['home_AdjO_rolling_10'] - df['home_AdjD_rolling_10']
    away_eff_margin_10 = df['away_AdjO_rolling_10'] - df['away_AdjD_rolling_10']
    features['home_momentum'] = features['home_eff_margin_5'] - home_eff_margin_10
    features['away_momentum'] = features['away_eff_margin_5'] - away_eff_margin_10
    
    # G-score differential
    features['g_score_diff_5'] = df['home_g_score_rolling_5'] - df['away_g_score_rolling_5']
    
    # SOS differential
    features['sos_diff'] = df['home_sos'] - df['away_sos']
    features['elite_sos_diff'] = df['home_elite_sos'] - df['away_elite_sos']
    
    # === OTHER DIFFERENTIALS ===
    features['tempo_diff'] = df['home_tempo'] - df['away_tempo']
    features['experience_diff'] = df['home_experience'] - df['away_experience']
    features['ft_pct_diff'] = df['home_ft_pct'] - df['away_ft_pct']
    features['barthag_diff'] = df['home_barthag'] - df['away_barthag']
    
    # === RANK DIFFERENTIALS (strongest signal found: +16 for def_rank, +15 for off_rank) ===
    # Lower rank = better team, so negative diff = home is better
    features['off_rank_diff'] = df['home_off_rank'] - df['away_off_rank']
    features['def_rank_diff'] = df['home_def_rank'] - df['away_def_rank']
    features['combined_rank_diff'] = features['off_rank_diff'] + features['def_rank_diff']
    
    # === INTERACTION FEATURES ===
    features['quality_x_home'] = features['quality_diff'] * features['home_court']
    features['matchup_x_home'] = features['matchup_diff'] * features['home_court']
    features['rank_x_home'] = features['combined_rank_diff'] * features['home_court']
    
    return features


def get_feature_list():
    """Get list of features for winner prediction."""
    return [
        # Context
        'is_neutral', 'is_conference', 'home_court',
        # Season differentials (strongest signals)
        'quality_diff', 'quality_diff_sqrt', 'net_eff_diff', 'talent_diff',
        'adj_off_diff', 'adj_def_diff', 'q12_pct_diff', 'q1_wins_diff',
        # Rank differentials
        'off_rank_diff', 'def_rank_diff', 'combined_rank_diff',
        # Danger zone indicators
        'is_danger_zone', 'is_mismatch',
        # Variance / volatility features
        'vol3_sum', 'vol3_diff',
        'to_pressure_diff',
        'expected_tempo', 'low_poss_flag', 'quality_x_low_poss',
        'floor_diff', 'ceiling_diff',
        # 3P features
        'home_3p_matchup', 'away_3p_matchup', 'threep_matchup_diff',
        'threep_off_diff', 'threep_matchup_danger',
        # Matchups
        'home_off_vs_away_def', 'away_off_vs_home_def', 'matchup_diff',
        # Four factors
        'efg_off_diff', 'efg_def_diff', 'tov_off_diff', 'tov_def_diff',
        'oreb_diff', 'dreb_diff', 'ftr_off_diff', 'ftr_def_diff',
        # Rebounding & ball movement
        'reb_battle', 'assist_diff', 'block_diff', 'tov_battle',
        # Conditional features (matter more in mismatches)
        'tov_battle_mismatch', 'oreb_diff_mismatch',
        # Rolling form
        'home_eff_margin_5', 'away_eff_margin_5', 'eff_margin_diff_5',
        'home_momentum', 'away_momentum', 'g_score_diff_5', 'sos_diff', 'elite_sos_diff',
        # Other
        'tempo_diff', 'experience_diff', 'ft_pct_diff', 'barthag_diff',
        # Interactions
        'quality_x_home', 'matchup_x_home', 'rank_x_home',
    ]


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model for winner prediction."""
    params = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def main():
    print('=' * 60)
    print('V4 WINNER-ONLY MODEL TRAINING')
    print('Optimized for Moneyline Prediction')
    print('=' * 60)
    
    # Load data
    print('\nLoading team data...')
    team_lookup = load_team_data()
    print(f'  Teams: {len(team_lookup)}')
    
    df = load_and_prepare_data(GAME_DATA_PATH, team_lookup)
    
    # Engineer features
    print('\nEngineering features...')
    features_df = engineer_features(df)
    features_df = features_df.fillna(features_df.median())
    
    feature_list = get_feature_list()
    print(f'  Features: {len(feature_list)}')
    
    # Target: spread (positive = home wins)
    y = df['home_score'] - df['away_score']
    
    # Remove missing
    valid = ~y.isna()
    features_df = features_df[valid]
    y = y[valid]
    
    print(f'\nTraining samples: {len(y)}')
    
    # Prepare data
    X = features_df[feature_list]
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print('\n' + '-' * 40)
    print('Training WINNER model...')
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_pred = model.predict(X_val)
    
    # Winner accuracy (main metric)
    winner_acc = np.mean((val_pred > 0) == (y_val.values > 0))
    print(f'  Winner Accuracy: {winner_acc:.1%}')
    
    # Confidence calibration
    confident_mask = np.abs(val_pred) > 5  # Confident predictions
    if confident_mask.sum() > 0:
        confident_acc = np.mean((val_pred[confident_mask] > 0) == (y_val.values[confident_mask] > 0))
        print(f'  Confident (|spread| > 5) Accuracy: {confident_acc:.1%} ({confident_mask.sum()} games)')
    
    close_mask = np.abs(val_pred) <= 3  # Close predictions
    if close_mask.sum() > 0:
        close_acc = np.mean((val_pred[close_mask] > 0) == (y_val.values[close_mask] > 0))
        print(f'  Close (|spread| <= 3) Accuracy: {close_acc:.1%} ({close_mask.sum()} games)')
    
    # Feature importance
    print('\n' + '-' * 40)
    print('Top 15 Feature Importances:')
    importance = pd.DataFrame({
        'feature': feature_list,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # Save model
    print('\n' + '-' * 40)
    print('Saving model...')
    
    model.save_model(MODEL_PATH)
    print(f'  Model: {MODEL_PATH}')
    
    # Save config
    config = {
        'model_version': 'v4',
        'model_type': 'winner_only',
        'description': 'Moneyline-optimized winner prediction model',
        'features': feature_list,
        'winner_accuracy': float(winner_acc),
        'training_samples': len(y),
    }
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Config: {CONFIG_PATH}')
    
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'\nWinner Accuracy: {winner_acc:.1%}')


if __name__ == '__main__':
    main()