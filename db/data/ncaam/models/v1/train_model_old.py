#!/usr/bin/env python3
"""
NCAAM Score Predictor - Training Script

Reads processed game data, reshapes to per-team rows, trains XGBRegressor, saves model.

Usage:
    python train_model.py
"""

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'processed')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')
MODEL_OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'ncaam_score_predictor.json')

# Features to use (from team's perspective)
FEATURES = [
    'team_AdjO', 'team_AdjD', 'team_T',
    'team_OEFF', 'team_DEFF',
    'team_OeFG', 'team_DeFG',
    'team_OTO', 'team_DTO',
    'team_OReb', 'team_DReb',
    'team_OFTR', 'team_DFTR',
    'team_g_score',
    'team_AdjO_rolling_5', 'team_AdjO_rolling_10',
    'team_AdjD_rolling_5', 'team_AdjD_rolling_10',
    'team_T_rolling_5', 'team_T_rolling_10',
    'team_sos',
    'opp_AdjO', 'opp_AdjD', 'opp_T',
    'opp_OEFF', 'opp_DEFF',
    'opp_OeFG', 'opp_DeFG',
    'opp_OTO', 'opp_DTO',
    'opp_OReb', 'opp_DReb',
    'opp_OFTR', 'opp_DFTR',
    'opp_g_score',
    'opp_AdjO_rolling_5', 'opp_AdjO_rolling_10',
    'opp_AdjD_rolling_5', 'opp_AdjD_rolling_10',
    'opp_T_rolling_5', 'opp_T_rolling_10',
    'opp_sos',
    'is_home'
]


def reshape_game_data(df):
    """
    Reshape game data from 1 row per game to 2 rows per game (one per team).
    """
    rows = []
    
    for _, game in df.iterrows():
        # Away team row
        away_row = {
            'team_code': game['away_team'],
            'opp_code': game['home_team'],
            'team_AdjO': game['away_AdjO'],
            'team_AdjD': game['away_AdjD'],
            'team_T': game['away_T'],
            'team_OEFF': game['away_OEFF'],
            'team_DEFF': game['away_DEFF'],
            'team_OeFG': game['away_OeFG%'],
            'team_DeFG': game['away_DeFG%'],
            'team_OTO': game['away_OTO%'],
            'team_DTO': game['away_DTO%'],
            'team_OReb': game['away_OReb%'],
            'team_DReb': game['away_DReb%'],
            'team_OFTR': game['away_OFTR'],
            'team_DFTR': game['away_DFTR'],
            'team_g_score': game['away_g_score'],
            'team_AdjO_rolling_5': game['away_AdjO_rolling_5'],
            'team_AdjO_rolling_10': game['away_AdjO_rolling_10'],
            'team_AdjD_rolling_5': game['away_AdjD_rolling_5'],
            'team_AdjD_rolling_10': game['away_AdjD_rolling_10'],
            'team_T_rolling_5': game['away_T_rolling_5'],
            'team_T_rolling_10': game['away_T_rolling_10'],
            'team_sos': game['away_sos'],
            'opp_AdjO': game['home_AdjO'],
            'opp_AdjD': game['home_AdjD'],
            'opp_T': game['home_T'],
            'opp_OEFF': game['home_OEFF'],
            'opp_DEFF': game['home_DEFF'],
            'opp_OeFG': game['home_OeFG%'],
            'opp_DeFG': game['home_DeFG%'],
            'opp_OTO': game['home_OTO%'],
            'opp_DTO': game['home_DTO%'],
            'opp_OReb': game['home_OReb%'],
            'opp_DReb': game['home_DReb%'],
            'opp_OFTR': game['home_OFTR'],
            'opp_DFTR': game['home_DFTR'],
            'opp_g_score': game['home_g_score'],
            'opp_AdjO_rolling_5': game['home_AdjO_rolling_5'],
            'opp_AdjO_rolling_10': game['home_AdjO_rolling_10'],
            'opp_AdjD_rolling_5': game['home_AdjD_rolling_5'],
            'opp_AdjD_rolling_10': game['home_AdjD_rolling_10'],
            'opp_T_rolling_5': game['home_T_rolling_5'],
            'opp_T_rolling_10': game['home_T_rolling_10'],
            'opp_sos': game['home_sos'],
            'is_home': 0,
            'score': game['away_score']
        }
        rows.append(away_row)
        
        # Home team row
        home_row = {
            'team_code': game['home_team'],
            'opp_code': game['away_team'],
            'team_AdjO': game['home_AdjO'],
            'team_AdjD': game['home_AdjD'],
            'team_T': game['home_T'],
            'team_OEFF': game['home_OEFF'],
            'team_DEFF': game['home_DEFF'],
            'team_OeFG': game['home_OeFG%'],
            'team_DeFG': game['home_DeFG%'],
            'team_OTO': game['home_OTO%'],
            'team_DTO': game['home_DTO%'],
            'team_OReb': game['home_OReb%'],
            'team_DReb': game['home_DReb%'],
            'team_OFTR': game['home_OFTR'],
            'team_DFTR': game['home_DFTR'],
            'team_g_score': game['home_g_score'],
            'team_AdjO_rolling_5': game['home_AdjO_rolling_5'],
            'team_AdjO_rolling_10': game['home_AdjO_rolling_10'],
            'team_AdjD_rolling_5': game['home_AdjD_rolling_5'],
            'team_AdjD_rolling_10': game['home_AdjD_rolling_10'],
            'team_T_rolling_5': game['home_T_rolling_5'],
            'team_T_rolling_10': game['home_T_rolling_10'],
            'team_sos': game['home_sos'],
            'opp_AdjO': game['away_AdjO'],
            'opp_AdjD': game['away_AdjD'],
            'opp_T': game['away_T'],
            'opp_OEFF': game['away_OEFF'],
            'opp_DEFF': game['away_DEFF'],
            'opp_OeFG': game['away_OeFG%'],
            'opp_DeFG': game['away_DeFG%'],
            'opp_OTO': game['away_OTO%'],
            'opp_DTO': game['away_DTO%'],
            'opp_OReb': game['away_OReb%'],
            'opp_DReb': game['away_DReb%'],
            'opp_OFTR': game['away_OFTR'],
            'opp_DFTR': game['away_DFTR'],
            'opp_g_score': game['away_g_score'],
            'opp_AdjO_rolling_5': game['away_AdjO_rolling_5'],
            'opp_AdjO_rolling_10': game['away_AdjO_rolling_10'],
            'opp_AdjD_rolling_5': game['away_AdjD_rolling_5'],
            'opp_AdjD_rolling_10': game['away_AdjD_rolling_10'],
            'opp_T_rolling_5': game['away_T_rolling_5'],
            'opp_T_rolling_10': game['away_T_rolling_10'],
            'opp_sos': game['away_sos'],
            'is_home': 1,
            'score': game['home_score']
        }
        rows.append(home_row)
    
    return pd.DataFrame(rows)


def train():
    print('=' * 60)
    print('Loading game data...')
    print('=' * 60)
    
    game_data = pd.read_csv(GAME_DATA_PATH)
    print(f'  Loaded {len(game_data)} games')
    
    print('\nReshaping to per-team rows...')
    training_data = reshape_game_data(game_data)
    print(f'  Created {len(training_data)} training rows')
    
    # Prepare features and target
    X = training_data[FEATURES]
    y = training_data['score']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f'\nTraining set: {len(X_train)} rows')
    print(f'Test set: {len(X_test)} rows')
    
    print('\n' + '=' * 60)
    print('Training XGBRegressor...')
    print('=' * 60)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'\nTest MAE: {mae:.2f} points')
    
    # Save model
    print('\n' + '=' * 60)
    print('Saving model...')
    print('=' * 60)
    
    model.save_model(MODEL_OUTPUT_PATH)
    print(f'  Saved to: {MODEL_OUTPUT_PATH}')
    
    print('\nDone!')


if __name__ == '__main__':
    train()