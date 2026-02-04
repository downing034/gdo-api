import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'processed')
GAME_DATA_PATH = os.path.join(DATA_DIR, 'base_model_game_data_with_rolling.csv')
SPREAD_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_spread_model.json')
TOTAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ncaam_total_model.json')


def load_and_prepare_data(filepath):
    """Load game data and create target variables."""
    print('Loading game data...')
    df = pd.read_csv(filepath)
    print(f'  Loaded {len(df)} games')
    
    # ==========================================================================
    # FILTER: Exclude early season games (before teams have settled)
    # ==========================================================================
    SEASON_START_CUTOFF = '11-20'  # MM-DD format, adjust as needed
    
    # Parse date and filter
    df['date'] = pd.to_datetime(df['date'])
    
    # Build cutoff dates for each season
    # Season column is like '24_25' or '25_26'
    def get_cutoff_date(row):
        season = row['season']
        if pd.isna(season):
            return pd.Timestamp('1900-01-01')  # Keep if no season
        # '24_25' means 2024-25 season, starts in fall 2024
        start_year = int('20' + season.split('_')[0])
        return pd.Timestamp(f'{start_year}-{SEASON_START_CUTOFF}')
    
    df['cutoff_date'] = df.apply(get_cutoff_date, axis=1)
    
    before_filter = len(df)
    df = df[df['date'] >= df['cutoff_date']]
    after_filter = len(df)
    
    print(f'  Filtered early season games (before {SEASON_START_CUTOFF}): {before_filter} → {after_filter} ({before_filter - after_filter} removed)')
    
    # Drop helper column
    df = df.drop(columns=['cutoff_date'])
    
    # Create target variables
    df['spread'] = df['home_score'] - df['away_score']  # positive = home win
    df['total'] = df['home_score'] + df['away_score']
    
    return df


def engineer_features(df):
    """
    Create features based on research findings:
    - Efficiency margin differentials (most predictive)
    - Four Factors differentials (eFG most important, then TOV, OReb, FTR)
    - Home court advantage (adjusted for neutral/conference)
    - Rest differential
    - SOS differential
    """
    print('Engineering features...')
    
    features = pd.DataFrame(index=df.index)
    
    # ==========================================================================
    # EFFICIENCY MARGIN DIFFERENTIAL (Research: most predictive single feature)
    # ==========================================================================
    # Rolling efficiency margins for each team
    features['home_eff_margin_5'] = df['home_AdjO_rolling_5'] - df['home_AdjD_rolling_5']
    features['away_eff_margin_5'] = df['away_AdjO_rolling_5'] - df['away_AdjD_rolling_5']
    features['home_eff_margin_10'] = df['home_AdjO_rolling_10'] - df['home_AdjD_rolling_10']
    features['away_eff_margin_10'] = df['away_AdjO_rolling_10'] - df['away_AdjD_rolling_10']
    
    # Differential (home - away)
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    features['eff_margin_diff_10'] = features['home_eff_margin_10'] - features['away_eff_margin_10']
    
    # ==========================================================================
    # FOUR FACTORS DIFFERENTIALS (Research: eFG >> TOV ≈ OReb >> FTR)
    # ==========================================================================
    
    # eFG% differential (most important of Four Factors)
    # Offensive comparison: whose offense shoots better
    features['eFG_off_diff_5'] = df['home_eFG_off_rolling_5'] - df['away_eFG_off_rolling_5']
    features['eFG_off_diff_10'] = df['home_eFG_off_rolling_10'] - df['away_eFG_off_rolling_10']
    
    # Defensive comparison: whose defense limits shooting better (lower is better for defense)
    features['eFG_def_diff_5'] = df['away_eFG_def_rolling_5'] - df['home_eFG_def_rolling_5']  # flipped so positive = home advantage
    features['eFG_def_diff_10'] = df['away_eFG_def_rolling_10'] - df['home_eFG_def_rolling_10']
    
    # Turnover% differential (lower is better for offense)
    features['TOV_off_diff_5'] = df['away_TOV_off_rolling_5'] - df['home_TOV_off_rolling_5']  # flipped so positive = home advantage
    features['TOV_off_diff_10'] = df['away_TOV_off_rolling_10'] - df['home_TOV_off_rolling_10']
    
    # Defensive turnover forcing (higher is better for defense)
    features['TOV_def_diff_5'] = df['home_TOV_def_rolling_5'] - df['away_TOV_def_rolling_5']
    features['TOV_def_diff_10'] = df['home_TOV_def_rolling_10'] - df['away_TOV_def_rolling_10']
    
    # Offensive rebounding differential
    features['OReb_diff_5'] = df['home_OReb_rolling_5'] - df['away_OReb_rolling_5']
    features['OReb_diff_10'] = df['home_OReb_rolling_10'] - df['away_OReb_rolling_10']
    
    # Defensive rebounding differential (higher DReb = fewer opponent OReb)
    features['DReb_diff_5'] = df['home_DReb_rolling_5'] - df['away_DReb_rolling_5']
    features['DReb_diff_10'] = df['home_DReb_rolling_10'] - df['away_DReb_rolling_10']
    
    # Free throw rate differential
    features['FTR_off_diff_5'] = df['home_FTR_off_rolling_5'] - df['away_FTR_off_rolling_5']
    features['FTR_off_diff_10'] = df['home_FTR_off_rolling_10'] - df['away_FTR_off_rolling_10']
    
    features['FTR_def_diff_5'] = df['away_FTR_def_rolling_5'] - df['home_FTR_def_rolling_5']  # flipped
    features['FTR_def_diff_10'] = df['away_FTR_def_rolling_10'] - df['home_FTR_def_rolling_10']
    
    # ==========================================================================
    # HOME COURT ADVANTAGE (Research: ~3.5 pts, reduced for conference games)
    # ==========================================================================
    # Base home advantage: 0 for neutral, ~3.5 otherwise
    features['is_neutral'] = (df['venue'] == 'N').astype(int)
    features['is_conference_game'] = (df['home_conf'] == df['away_conf']).astype(int)
    
    # Home court value: 0 if neutral, 3.0 if conference, 3.5 if non-conference
    features['home_court_advantage'] = np.where(
        df['venue'] == 'N',
        0.0,
        np.where(
            df['home_conf'] == df['away_conf'],
            3.0,  # conference game
            3.5   # non-conference
        )
    )
    
    # ==========================================================================
    # REST DIFFERENTIAL (Research: minor but real effect)
    # ==========================================================================
    features['rest_diff'] = df['home_days_rest'] - df['away_days_rest']
    
    # Cap extreme values (team coming off long break vs back-to-back)
    features['rest_diff'] = features['rest_diff'].clip(-7, 7)
    
    # ==========================================================================
    # STRENGTH OF SCHEDULE (Research: important for context)
    # ==========================================================================
    features['sos_diff'] = df['home_sos'] - df['away_sos']
    
    # ==========================================================================
    # TEMPO (for total points model primarily)
    # ==========================================================================
    features['home_tempo_5'] = df['home_T_rolling_5']
    features['away_tempo_5'] = df['away_T_rolling_5']
    features['avg_tempo_5'] = (df['home_T_rolling_5'] + df['away_T_rolling_5']) / 2
    features['tempo_diff_5'] = df['home_T_rolling_5'] - df['away_T_rolling_5']
    
    # ==========================================================================
    # GAME SCORE ROLLING (overall team performance indicator)
    # ==========================================================================
    features['g_score_diff_5'] = df['home_g_score_rolling_5'] - df['away_g_score_rolling_5']
    features['g_score_diff_10'] = df['home_g_score_rolling_10'] - df['away_g_score_rolling_10']
    
    # ==========================================================================
    # COMBINED OFFENSIVE/DEFENSIVE STRENGTH (for total points model)
    # ==========================================================================
    features['combined_AdjO_5'] = df['home_AdjO_rolling_5'] + df['away_AdjO_rolling_5']
    features['combined_AdjD_5'] = df['home_AdjD_rolling_5'] + df['away_AdjD_rolling_5']
    
    print(f'  Created {len(features.columns)} features')
    
    return features


def get_spread_features():
    """Features specifically for spread prediction."""
    return [
        # Efficiency margins (most important)
        'eff_margin_diff_5',
        'eff_margin_diff_10',
        
        # Four Factors - offensive
        'eFG_off_diff_5',
        'eFG_off_diff_10',
        'TOV_off_diff_5',
        'TOV_off_diff_10',
        'OReb_diff_5',
        'OReb_diff_10',
        'FTR_off_diff_5',
        'FTR_off_diff_10',
        
        # Four Factors - defensive
        'eFG_def_diff_5',
        'eFG_def_diff_10',
        'TOV_def_diff_5',
        'TOV_def_diff_10',
        'DReb_diff_5',
        'DReb_diff_10',
        'FTR_def_diff_5',
        'FTR_def_diff_10',
        
        # Home court
        'home_court_advantage',
        'is_neutral',
        'is_conference_game',
        
        # Rest
        'rest_diff',
        
        # SOS
        'sos_diff',
        
        # Game score
        'g_score_diff_5',
        'g_score_diff_10',
    ]


def get_total_features():
    """Features specifically for total points prediction."""
    return [
        # Tempo (most important for totals)
        'avg_tempo_5',
        'home_tempo_5',
        'away_tempo_5',
        
        # Combined offensive strength
        'combined_AdjO_5',
        'combined_AdjD_5',
        
        # Individual efficiency margins (game pace context)
        'home_eff_margin_5',
        'away_eff_margin_5',
        
        # Shooting efficiency (affects scoring)
        'eFG_off_diff_5',
        'eFG_def_diff_5',
        
        # Turnover rates (possessions lost/gained)
        'TOV_off_diff_5',
        'TOV_def_diff_5',
        
        # Pace-related
        'tempo_diff_5',
        
        # Home court (slight effect on total)
        'is_neutral',
    ]


def train_model(X_train, y_train, X_test, y_test, model_name):
    """Train XGBoost model with reasonable defaults."""
    print(f'\nTraining {model_name}...')
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f'  Test MAE: {mae:.2f}')
    print(f'  Test RMSE: {rmse:.2f}')
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f'\n  Top 10 features:')
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, mae, rmse


def main():
    print('=' * 60)
    print('NCAAM Basketball Model Training')
    print('=' * 60)
    
    # Load data
    df = load_and_prepare_data(GAME_DATA_PATH)
    
    # Engineer features
    features = engineer_features(df)
    
    # Handle missing values (first games of season)
    print(f'\nMissing values before fillna: {features.isnull().sum().sum()}')
    features = features.fillna(features.median())
    print(f'Missing values after fillna: {features.isnull().sum().sum()}')
    
    # Get targets
    spread = df['spread']
    total = df['total']
    
    # Split data (time-based would be better, but random for now)
    # Using same split for both models for consistency
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42
    )
    
    print(f'\nTraining set: {len(train_idx)} games')
    print(f'Test set: {len(test_idx)} games')
    
    # ==========================================================================
    # SPREAD MODEL
    # ==========================================================================
    print('\n' + '=' * 60)
    print('SPREAD MODEL (predicts home_score - away_score)')
    print('=' * 60)
    
    spread_features = get_spread_features()
    X_spread = features[spread_features]
    
    spread_model, spread_mae, spread_rmse = train_model(
        X_spread.loc[train_idx],
        spread.loc[train_idx],
        X_spread.loc[test_idx],
        spread.loc[test_idx],
        'Spread Model'
    )
    
    # ==========================================================================
    # TOTAL MODEL
    # ==========================================================================
    print('\n' + '=' * 60)
    print('TOTAL MODEL (predicts home_score + away_score)')
    print('=' * 60)
    
    total_features = get_total_features()
    X_total = features[total_features]
    
    total_model, total_mae, total_rmse = train_model(
        X_total.loc[train_idx],
        total.loc[train_idx],
        X_total.loc[test_idx],
        total.loc[test_idx],
        'Total Model'
    )
    
    # ==========================================================================
    # SAVE MODELS
    # ==========================================================================
    print('\n' + '=' * 60)
    print('Saving models...')
    print('=' * 60)
    
    spread_model.save_model(SPREAD_MODEL_PATH)
    print(f'  Spread model saved to: {SPREAD_MODEL_PATH}')
    
    total_model.save_model(TOTAL_MODEL_PATH)
    print(f'  Total model saved to: {TOTAL_MODEL_PATH}')
    
    # Save feature lists for prediction
    config = {
        'spread_features': spread_features,
        'total_features': total_features,
        'spread_mae': spread_mae,
        'spread_rmse': spread_rmse,
        'total_mae': total_mae,
        'total_rmse': total_rmse,
    }
    
    config_path = os.path.join(SCRIPT_DIR, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Config saved to: {config_path}')





    print('=' * 60)
    print('NCAAM Basketball Model Training')
    print('=' * 60)
    
    # Load data
    df = load_and_prepare_data(GAME_DATA_PATH)
    
    # ==========================================================================
    # DATASET STATISTICS
    # ==========================================================================
    print('\n' + '=' * 60)
    print('DATASET STATISTICS')
    print('=' * 60)
    
    totals = df['home_score'] + df['away_score']
    spreads = df['home_score'] - df['away_score']
    
    print(f'  Total points - Mean: {totals.mean():.1f}, Median: {totals.median():.1f}, Std: {totals.std():.1f}')
    print(f'  Spread - Mean: {spreads.mean():.1f}, Median: {spreads.median():.1f}, Std: {spreads.std():.1f}')
    print(f'  Home score - Mean: {df["home_score"].mean():.1f}')
    print(f'  Away score - Mean: {df["away_score"].mean():.1f}')
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Spread Model MAE: {spread_mae:.2f} points')
    print(f'Total Model MAE: {total_mae:.2f} points')
    print('\nTo get predicted scores:')
    print('  home_score = (total + spread) / 2')
    print('  away_score = (total - spread) / 2')
    print('\nDone!')


if __name__ == '__main__':
    main()