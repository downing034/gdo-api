"""
V3 Training Script: Dual Score Models (Home + Away)

Two separate models:
- Home score model (36 features)
- Away score model (34 features)

Spread = home_score - away_score
Total = home_score + away_score

Results from GA optimization:
- 311k unique combinations tested
- Home MAE: 8.36, Away MAE: 8.10
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import json

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'processed')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')
TEAM_DATA_PATH = os.path.join(DATA_DIR, 'ncaam_team_data_final.csv')
HOME_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_home_score_model.json')
AWAY_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_away_score_model.json')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'model_config.json')


def safe_float(val, default):
    try:
        return float(val) if val and str(val).strip() != '' else default
    except:
        return default


def safe_divide(a, b, default=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, default)
        result = np.where(np.isfinite(result), result, default)
    return result


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


def load_and_prepare_data(filepath, team_lookup):
    """Load game data and add team season stats."""
    print('Loading game data...')
    df = pd.read_csv(filepath, keep_default_na=False, na_values=[''])
    print(f'  Loaded {len(df)} games')
    
    # Filter early season
    df['date'] = pd.to_datetime(df['date'])
    
    def get_cutoff_date(row):
        season = row['season']
        if pd.isna(season):
            return pd.Timestamp('1900-01-01')
        start_year = int('20' + season.split('_')[0])
        return pd.Timestamp(f'{start_year}-11-20')
    
    df['cutoff_date'] = df.apply(get_cutoff_date, axis=1)
    before_filter = len(df)
    df = df[df['date'] >= df['cutoff_date']]
    print(f'  Filtered: {before_filter} → {len(df)}')
    df = df.drop(columns=['cutoff_date'])
    
    # Add team season stats
    for prefix, col in [('home', 'home_team'), ('away', 'away_team')]:
        for key in ['adj_off', 'adj_def', 'off_rank', 'def_rank', 'barthag_rank', 'barthag',
                    'efg_off', 'efg_def', 'ftr_off', 'ftr_def', 'tov_off', 'tov_def',
                    'oreb', 'dreb', '3p_off', '3p_def', '2p_off', '2p_def', 'ft_pct',
                    'tempo', 'experience', 'talent', 'block_def', 'assist_off', '3p_rate_off',
                    'quality_score', 'weighted_quality']:
            default = 0 if key in ['quality_score', 'weighted_quality', 'talent'] else 50
            df[f'{prefix}_{key}'] = df[col].apply(lambda x: team_lookup.get(x, {}).get(key, default))
    
    return df


def engineer_home_features(df):
    """Engineer features for home score prediction."""
    features = pd.DataFrame(index=df.index)
    
    # Quality matchup (home quality vs away quality)
    features['quality_matchup'] = df['home_weighted_quality'] - df['away_weighted_quality']
    features['quality_matchup_sqrt'] = np.sign(features['quality_matchup']) * np.sqrt(np.abs(features['quality_matchup']))
    features['quality_matchup_sq'] = features['quality_matchup'] ** 2
    
    # Context
    features['is_neutral'] = (df['venue'] == 'N').astype(float)
    features['is_conference'] = (df['home_conf'] == df['away_conf']).astype(float)
    features['home_court'] = np.where(
        features['is_neutral'] == 1, 0.0,
        np.where(features['is_conference'] == 1, 2.5, 3.5)
    )
    features['home_days_rest'] = np.clip(df['home_days_rest'], 0, 14)
    
    # Quality x context
    features['quality_x_home'] = features['quality_matchup'] * features['home_court']
    
    # Tempo features
    features['avg_tempo_5'] = (df['home_T_rolling_5'] + df['away_T_rolling_5']) / 2
    features['avg_tempo_10'] = (df['home_T_rolling_10'] + df['away_T_rolling_10']) / 2
    features['avg_tempo_season'] = (df['home_tempo'] + df['away_tempo']) / 2
    
    # Home offensive stats
    features['home_AdjO_10'] = df['home_AdjO_rolling_10']
    features['home_eFG_off_10'] = df['home_eFG_off_rolling_10']
    features['home_tempo_10'] = df['home_T_rolling_10']
    features['home_OReb_5'] = df['home_OReb_rolling_5']
    features['home_OReb_10'] = df['home_OReb_rolling_10']
    features['home_FTR_off_10'] = df['home_FTR_off_rolling_10']
    
    # Home season stats
    features['home_off_rank'] = df['home_off_rank']
    features['home_2p_off'] = df['home_2p_off']
    features['home_3p_off'] = df['home_3p_off']
    features['home_3p_rate_off'] = df['home_3p_rate_off']
    features['home_oreb'] = df['home_oreb']
    features['home_talent'] = df['home_talent']
    features['home_tempo'] = df['home_tempo']
    features['home_weighted_quality'] = df['home_weighted_quality']
    
    # Away defensive stats (opponent defense)
    features['away_AdjD_5'] = df['away_AdjD_rolling_5']
    features['away_DReb_5'] = df['away_DReb_rolling_5']
    features['away_FTR_def_10'] = df['away_FTR_def_rolling_10']
    
    # Away season defensive stats
    features['away_adj_def'] = df['away_adj_def']
    features['away_2p_def'] = df['away_2p_def']
    features['away_3p_def'] = df['away_3p_def']
    features['away_dreb'] = df['away_dreb']
    
    # Matchup features
    features['off_vs_def_eff_10'] = df['home_AdjO_rolling_10'] - df['away_AdjD_rolling_10']
    off_vs_def_eff_5 = df['home_AdjO_rolling_5'] - df['away_AdjD_rolling_5']
    features['off_vs_def_eff_5_sqrt'] = np.sign(off_vs_def_eff_5) * np.sqrt(np.abs(off_vs_def_eff_5))
    
    # Tempo interactions
    features['off_eff_x_tempo'] = df['home_AdjO_rolling_5'] * features['avg_tempo_5'] / 70
    
    # Context interactions
    features['matchup_x_home'] = features['off_vs_def_eff_10'] * features['home_court']
    features['matchup_x_conf'] = features['off_vs_def_eff_10'] * features['is_conference']
    
    # === NEW: Form/Momentum features (V1-style, helps moneyline) ===
    # Team efficiency margins (who's playing better overall)
    features['home_eff_margin_5'] = df['home_AdjO_rolling_5'] - df['home_AdjD_rolling_5']
    features['away_eff_margin_5'] = df['away_AdjO_rolling_5'] - df['away_AdjD_rolling_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    
    # Momentum (is team improving or declining?)
    home_eff_margin_10 = df['home_AdjO_rolling_10'] - df['home_AdjD_rolling_10']
    away_eff_margin_10 = df['away_AdjO_rolling_10'] - df['away_AdjD_rolling_10']
    features['home_momentum'] = features['home_eff_margin_5'] - home_eff_margin_10
    features['away_momentum'] = features['away_eff_margin_5'] - away_eff_margin_10
    
    # G-score diff (composite form metric from V1)
    features['g_score_diff_5'] = df['home_g_score_rolling_5'] - df['away_g_score_rolling_5']
    
    # SOS diff (strength of schedule)
    features['sos_diff'] = df['home_sos'] - df['away_sos']
    
    # === NEW: Defensive Matchup Adjustments ===
    # V3 under-predicts scoring vs bad defenses by ~2-3 pts
    # 3P defense weakness (positive = bad defense, allows more 3s)
    # Average 3P def is ~34%, so measure deviation from that
    features['opp_3p_def_weakness'] = df['away_3p_def'] - 34.0
    features['opp_efg_def_weakness'] = df['away_efg_def'] - 51.0  # Avg eFG def ~51%
    
    # Scoring boost potential (offense vs weak defense)
    features['home_3p_vs_def'] = df['home_3p_off'] - df['away_3p_def']  # Positive = advantage
    features['home_efg_vs_def'] = df['home_efg_off'] - df['away_efg_def']
    
    # === NEW: Close Game Tiebreaker Features ===
    # FT% advantage (matters in close games - 11.5% swing found)
    features['ft_pct_diff'] = df['home_ft_pct'] - df['away_ft_pct']
    
    # Net matchup differential (home offense advantage vs away offense advantage)
    features['matchup_diff'] = (df['home_adj_off'] - df['away_adj_def']) - (df['away_adj_off'] - df['home_adj_def'])
    
    # 3P matchup differential
    features['threep_matchup_diff'] = (df['home_3p_off'] - df['away_3p_def']) - (df['away_3p_off'] - df['home_3p_def'])
    
    # === NEW: Rebounding & Ball Movement Features ===
    # Rebounding battle: net second-chance advantage (-0.98 signal in wrong preds)
    # OREB% vs (100 - opponent DREB%) = net offensive rebounding edge
    features['reb_battle'] = (df['home_oreb'] - (100 - df['away_dreb'])) - (df['away_oreb'] - (100 - df['home_dreb']))
    
    # Assist differential: ball movement / team play (-0.74 signal)
    features['assist_diff'] = df['home_assist_off'] - df['away_assist_off']
    
    # Block differential: rim protection (-0.45 signal)
    features['block_diff'] = df['home_block_def'] - df['away_block_def']
    
    return features


def engineer_away_features(df):
    """Engineer features for away score prediction."""
    features = pd.DataFrame(index=df.index)
    
    # Quality matchup (away perspective: away quality - home quality for away scoring)
    quality_matchup = df['away_weighted_quality'] - df['home_weighted_quality']
    features['quality_matchup_sqrt'] = np.sign(quality_matchup) * np.sqrt(np.abs(quality_matchup))
    
    # Context
    features['is_conference'] = (df['home_conf'] == df['away_conf']).astype(float)
    features['is_neutral'] = (df['venue'] == 'N').astype(float)
    features['home_court'] = np.where(
        features['is_neutral'] == 1, 0.0,
        np.where(features['is_conference'] == 1, -2.5, -3.5)  # Negative for away team
    )
    features['home_days_rest'] = np.clip(df['home_days_rest'], 0, 14)  # Affects away scoring
    
    # Tempo features
    features['avg_tempo_5'] = (df['home_T_rolling_5'] + df['away_T_rolling_5']) / 2
    features['avg_tempo_10'] = (df['home_T_rolling_10'] + df['away_T_rolling_10']) / 2
    
    # Away offensive stats
    features['away_AdjO_10'] = df['away_AdjO_rolling_10']
    features['away_eFG_off_5'] = df['away_eFG_off_rolling_5']
    features['away_eFG_off_10'] = df['away_eFG_off_rolling_10']
    features['away_tempo_5'] = df['away_T_rolling_5']
    features['away_TOV_off_10'] = df['away_TOV_off_rolling_10']
    features['away_OReb_5'] = df['away_OReb_rolling_5']
    features['away_OReb_10'] = df['away_OReb_rolling_10']
    features['away_FTR_off_10'] = df['away_FTR_off_rolling_10']
    features['away_g_score_10'] = df['away_g_score_rolling_10']
    
    # Away season stats
    features['away_efg_off'] = df['away_efg_off']
    features['away_2p_off'] = df['away_2p_off']
    features['away_3p_rate_off'] = df['away_3p_rate_off']
    features['away_weighted_quality'] = df['away_weighted_quality']
    
    # Home defensive stats (opponent defense)
    features['home_AdjD_10'] = df['home_AdjD_rolling_10']
    features['home_eFG_def_10'] = df['home_eFG_def_rolling_10']
    features['home_DReb_10'] = df['home_DReb_rolling_10']
    features['home_FTR_def_10'] = df['home_FTR_def_rolling_10']
    
    # Home season defensive stats
    features['home_def_rank'] = df['home_def_rank']
    features['home_2p_def'] = df['home_2p_def']
    features['home_tov_def'] = df['home_tov_def']
    
    # Matchup features
    features['off_vs_def_eff_10'] = df['away_AdjO_rolling_10'] - df['home_AdjD_rolling_10']
    features['off_vs_def_eff_10_sq'] = features['off_vs_def_eff_10'] ** 2
    features['eFG_matchup_5'] = df['away_eFG_off_rolling_5'] - df['home_eFG_def_rolling_5']
    features['eFG_matchup_10'] = df['away_eFG_off_rolling_10'] - df['home_eFG_def_rolling_10']
    features['FTR_matchup_5'] = df['away_FTR_off_rolling_5'] - df['home_FTR_def_rolling_5']
    features['FTR_matchup_10'] = df['away_FTR_off_rolling_10'] - df['home_FTR_def_rolling_10']
    
    # Tempo interaction
    features['matchup_x_tempo'] = features['off_vs_def_eff_10'] * features['avg_tempo_10'] / 70
    
    # Context interaction
    features['matchup_x_home'] = features['off_vs_def_eff_10'] * features['home_court']
    
    # === NEW: Form/Momentum features (V1-style, helps moneyline) ===
    # Team efficiency margins (who's playing better overall)
    features['home_eff_margin_5'] = df['home_AdjO_rolling_5'] - df['home_AdjD_rolling_5']
    features['away_eff_margin_5'] = df['away_AdjO_rolling_5'] - df['away_AdjD_rolling_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    
    # Momentum (is team improving or declining?)
    home_eff_margin_10 = df['home_AdjO_rolling_10'] - df['home_AdjD_rolling_10']
    away_eff_margin_10 = df['away_AdjO_rolling_10'] - df['away_AdjD_rolling_10']
    features['home_momentum'] = features['home_eff_margin_5'] - home_eff_margin_10
    features['away_momentum'] = features['away_eff_margin_5'] - away_eff_margin_10
    
    # G-score diff (composite form metric from V1)
    features['g_score_diff_5'] = df['home_g_score_rolling_5'] - df['away_g_score_rolling_5']
    
    # SOS diff (strength of schedule)
    features['sos_diff'] = df['home_sos'] - df['away_sos']
    
    # === NEW: Defensive Matchup Adjustments ===
    # Away team faces home defense - measure weakness
    features['opp_3p_def_weakness'] = df['home_3p_def'] - 34.0
    features['opp_efg_def_weakness'] = df['home_efg_def'] - 51.0
    
    # Scoring boost potential (away offense vs home defense)
    features['away_3p_vs_def'] = df['away_3p_off'] - df['home_3p_def']
    features['away_efg_vs_def'] = df['away_efg_off'] - df['home_efg_def']
    
    # === NEW: Close Game Tiebreaker Features ===
    # FT% advantage (matters in close games)
    features['ft_pct_diff'] = df['home_ft_pct'] - df['away_ft_pct']
    
    # Net matchup differential
    features['matchup_diff'] = (df['home_adj_off'] - df['away_adj_def']) - (df['away_adj_off'] - df['home_adj_def'])
    
    # 3P matchup differential
    features['threep_matchup_diff'] = (df['home_3p_off'] - df['away_3p_def']) - (df['away_3p_off'] - df['home_3p_def'])
    
    # === NEW: Rebounding & Ball Movement Features ===
    # Rebounding battle: net second-chance advantage
    features['reb_battle'] = (df['home_oreb'] - (100 - df['away_dreb'])) - (df['away_oreb'] - (100 - df['home_dreb']))
    
    # Assist differential: ball movement / team play
    features['assist_diff'] = df['home_assist_off'] - df['away_assist_off']
    
    # Block differential: rim protection
    features['block_diff'] = df['home_block_def'] - df['away_block_def']
    
    return features


def get_home_features():
    """GA-selected features for home score prediction + V1-style form features."""
    return [
        'quality_matchup', 'quality_x_home', 'quality_matchup_sqrt', 'off_eff_x_tempo',
        'avg_tempo_10', 'avg_tempo_5', 'home_AdjO_10', 'home_off_rank', 'home_2p_off',
        'home_tempo_10', 'home_eFG_off_10', 'home_weighted_quality', 'quality_matchup_sq',
        'avg_tempo_season', 'home_court', 'away_2p_def', 'home_talent', 'home_tempo',
        'away_adj_def', 'away_AdjD_5', 'is_neutral', 'home_3p_off', 'away_3p_def',
        'home_OReb_10', 'home_oreb', 'home_OReb_5', 'away_DReb_5', 'away_dreb',
        'matchup_x_home', 'off_vs_def_eff_10', 'off_vs_def_eff_5_sqrt', 'home_3p_rate_off',
        'home_FTR_off_10', 'away_FTR_def_10', 'matchup_x_conf', 'home_days_rest',
        # V1-style form/momentum features
        'home_eff_margin_5', 'away_eff_margin_5', 'eff_margin_diff_5',
        'home_momentum', 'away_momentum', 'g_score_diff_5', 'sos_diff',
        # Defensive matchup adjustments (V3 under-predicts vs bad defense)
        'opp_3p_def_weakness', 'opp_efg_def_weakness',
        'home_3p_vs_def', 'home_efg_vs_def',
        # Close game tiebreaker features
        'ft_pct_diff', 'matchup_diff', 'threep_matchup_diff',
        # Rebounding & ball movement features
        'reb_battle', 'assist_diff', 'block_diff'
    ]


def get_away_features():
    """GA-selected features for away score prediction + V1-style form features."""
    return [
        'quality_matchup_sqrt', 'away_AdjO_10', 'avg_tempo_10', 'avg_tempo_5',
        'away_g_score_10', 'away_efg_off', 'away_2p_off', 'away_eFG_off_10',
        'away_weighted_quality', 'away_tempo_5', 'away_TOV_off_10', 'away_eFG_off_5',
        'home_court', 'home_AdjD_10', 'home_eFG_def_10', 'home_2p_def', 'home_def_rank',
        'away_OReb_10', 'away_3p_rate_off', 'away_OReb_5', 'home_tov_def', 'home_DReb_10',
        'off_vs_def_eff_10', 'away_FTR_off_10', 'matchup_x_tempo', 'off_vs_def_eff_10_sq',
        'is_conference', 'eFG_matchup_10', 'matchup_x_home', 'eFG_matchup_5',
        'FTR_matchup_10', 'FTR_matchup_5', 'home_FTR_def_10', 'home_days_rest',
        # V1-style form/momentum features
        'home_eff_margin_5', 'away_eff_margin_5', 'eff_margin_diff_5',
        'home_momentum', 'away_momentum', 'g_score_diff_5', 'sos_diff',
        # Defensive matchup adjustments (V3 under-predicts vs bad defense)
        'opp_3p_def_weakness', 'opp_efg_def_weakness',
        'away_3p_vs_def', 'away_efg_vs_def',
        # Close game tiebreaker features
        'ft_pct_diff', 'matchup_diff', 'threep_matchup_diff',
        # Rebounding & ball movement features
        'reb_battle', 'assist_diff', 'block_diff'
    ]


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    params = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'early_stopping_rounds': 50,
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model


def main():
    print('=' * 60)
    print('V3 MODEL TRAINING (Dual Home/Away Score Models)')
    print('=' * 60)
    
    # Load data
    print('\nLoading team data...')
    team_lookup = load_team_data()
    print(f'  Teams: {len(team_lookup)}')
    
    df = load_and_prepare_data(GAME_DATA_PATH, team_lookup)
    
    # Engineer features
    print('\nEngineering features...')
    home_features_df = engineer_home_features(df)
    away_features_df = engineer_away_features(df)
    
    home_features_df = home_features_df.fillna(home_features_df.median())
    away_features_df = away_features_df.fillna(away_features_df.median())
    
    # Get feature lists
    home_feature_list = get_home_features()
    away_feature_list = get_away_features()
    
    print(f'  Home features: {len(home_feature_list)}')
    print(f'  Away features: {len(away_feature_list)}')
    
    # Targets
    y_home = df['home_score']
    y_away = df['away_score']
    
    # Remove missing
    valid = ~y_home.isna() & ~y_away.isna()
    home_features_df = home_features_df[valid]
    away_features_df = away_features_df[valid]
    y_home = y_home[valid]
    y_away = y_away[valid]
    
    print(f'\nTraining samples: {len(y_home)}')
    
    # Prepare data
    X_home = home_features_df[home_feature_list]
    X_away = away_features_df[away_feature_list]
    
    # Split
    X_home_train, X_home_val, y_home_train, y_home_val = train_test_split(
        X_home, y_home, test_size=0.2, random_state=42
    )
    X_away_train, X_away_val, y_away_train, y_away_val = train_test_split(
        X_away, y_away, test_size=0.2, random_state=42
    )
    
    # Train home score model
    print('\n' + '-' * 40)
    print('Training HOME SCORE model...')
    home_model = train_model(X_home_train, y_home_train, X_home_val, y_home_val)
    
    home_val_pred = home_model.predict(X_home_val)
    home_val_mae = mean_absolute_error(y_home_val, home_val_pred)
    print(f'  Val MAE: {home_val_mae:.2f}')
    
    # Train away score model
    print('\n' + '-' * 40)
    print('Training AWAY SCORE model...')
    away_model = train_model(X_away_train, y_away_train, X_away_val, y_away_val)
    
    away_val_pred = away_model.predict(X_away_val)
    away_val_mae = mean_absolute_error(y_away_val, away_val_pred)
    print(f'  Val MAE: {away_val_mae:.2f}')
    
    # Combined metrics
    print('\n' + '-' * 40)
    print('Combined metrics (validation):')
    spread_pred = home_val_pred - away_val_pred
    spread_actual = y_home_val.values - y_away_val.values
    spread_mae = mean_absolute_error(spread_actual, spread_pred)
    spread_acc = np.mean((spread_pred > 0) == (spread_actual > 0))
    
    total_pred = home_val_pred + away_val_pred
    total_actual = y_home_val.values + y_away_val.values
    total_mae = mean_absolute_error(total_actual, total_pred)
    
    print(f'  Spread MAE: {spread_mae:.2f}')
    print(f'  Spread Acc: {spread_acc:.1%}')
    print(f'  Total MAE: {total_mae:.2f}')
    
    # Feature importance
    print('\n' + '-' * 40)
    print('Top 10 HOME SCORE Feature Importances:')
    home_importance = pd.DataFrame({
        'feature': home_feature_list,
        'importance': home_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in home_importance.head(10).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.4f}")
    
    print('\nTop 10 AWAY SCORE Feature Importances:')
    away_importance = pd.DataFrame({
        'feature': away_feature_list,
        'importance': away_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in away_importance.head(10).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.4f}")
    
    # Save models
    print('\n' + '-' * 40)
    print('Saving models...')
    
    home_model.save_model(HOME_MODEL_PATH)
    print(f'  Home: {HOME_MODEL_PATH}')
    
    away_model.save_model(AWAY_MODEL_PATH)
    print(f'  Away: {AWAY_MODEL_PATH}')
    
    # Save config
    config = {
        'model_version': 'v3',
        'model_type': 'dual_score',
        'description': 'Separate home/away score models from 311k GA combinations',
        'home_features': home_feature_list,
        'away_features': away_feature_list,
        'home_mae': float(home_val_mae),
        'away_mae': float(away_val_mae),
        'spread_mae': float(spread_mae),
        'spread_acc': float(spread_acc),
        'total_mae': float(total_mae),
        'training_samples': len(y_home),
    }
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Config: {CONFIG_PATH}')
    
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'\nHome Score MAE: {home_val_mae:.2f}')
    print(f'Away Score MAE: {away_val_mae:.2f}')
    print(f'Spread MAE: {spread_mae:.2f}')
    print(f'Spread Acc: {spread_acc:.1%}')
    print(f'Total MAE: {total_mae:.2f}')


if __name__ == '__main__':
    main()