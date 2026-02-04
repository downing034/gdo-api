"""
Systematic Feature Discovery and Selection for NCAAM

This script:
1. Loads raw stats and known good features
2. Generates hundreds of candidate features (ratios, products, interactions)
3. Filters out highly correlated redundant features
4. Uses forward selection to find optimal feature subset
5. Compares to V1 baseline

The goal is to discover feature combinations we wouldn't think of manually.
"""

import os
import pandas as pd
import numpy as np
from itertools import combinations, product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
GAME_DATA_PATH = os.path.join(BASE_DIR, 'processed', 'base_model_game_data_with_rolling.csv')
TEAM_DATA_PATH = os.path.join(BASE_DIR, 'processed', 'ncaam_team_data_final.csv')


def load_data():
    """Load and prepare game data."""
    print("Loading data...")
    df = pd.read_csv(GAME_DATA_PATH, keep_default_na=False, na_values=[''])
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter early season
    def get_cutoff_date(row):
        season = row['season']
        if pd.isna(season):
            return pd.Timestamp('1900-01-01')
        start_year = int('20' + str(season).split('_')[0])
        return pd.Timestamp(f'{start_year}-11-20')
    
    df['cutoff_date'] = df.apply(get_cutoff_date, axis=1)
    df = df[df['date'] >= df['cutoff_date']]
    
    # Load team data for additional features
    teams = pd.read_csv(TEAM_DATA_PATH, keep_default_na=False, na_values=[''])
    team_lookup = {}
    for _, row in teams.iterrows():
        code = row.get('Team_Code')
        if code:
            team_lookup[code] = {
                'def_rank': safe_float(row.get('Adj. Def. Eff Rank'), 180),
                'off_rank': safe_float(row.get('Adj. Off. Eff Rank'), 180),
                'barthag_rank': safe_float(row.get('Team ID'), 180),
                '3p_off': safe_float(row.get('3P% Off'), 33),
                '3p_def': safe_float(row.get('3P% Def'), 33),
                '3p_off_rank': safe_float(row.get('3P% Off Rank'), 180),
            }
    
    # Add team data to games
    for prefix, col in [('home', 'home_team'), ('away', 'away_team')]:
        df[f'{prefix}_def_rank'] = df[col].apply(lambda x: team_lookup.get(x, {}).get('def_rank', 180))
        df[f'{prefix}_off_rank'] = df[col].apply(lambda x: team_lookup.get(x, {}).get('off_rank', 180))
        df[f'{prefix}_barthag_rank'] = df[col].apply(lambda x: team_lookup.get(x, {}).get('barthag_rank', 180))
        df[f'{prefix}_3p_off'] = df[col].apply(lambda x: team_lookup.get(x, {}).get('3p_off', 33))
        df[f'{prefix}_3p_def'] = df[col].apply(lambda x: team_lookup.get(x, {}).get('3p_def', 33))
    
    # Target
    df['margin'] = df['home_score'] - df['away_score']
    df['total'] = df['home_score'] + df['away_score']
    
    print(f"  Games: {len(df)}")
    return df


def safe_float(val, default):
    """Safely convert to float."""
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def create_base_features(df):
    """
    Create Layer 1 (raw) and Layer 2 (known good) features.
    Returns dict of feature arrays.
    """
    features = {}
    
    # ==========================================================================
    # LAYER 1: RAW STATS (building blocks)
    # ==========================================================================
    
    # Rolling efficiency stats
    for prefix in ['home', 'away']:
        features[f'{prefix}_AdjO_5'] = df[f'{prefix}_AdjO_rolling_5'].values
        features[f'{prefix}_AdjD_5'] = df[f'{prefix}_AdjD_rolling_5'].values
        features[f'{prefix}_AdjO_10'] = df[f'{prefix}_AdjO_rolling_10'].values
        features[f'{prefix}_AdjD_10'] = df[f'{prefix}_AdjD_rolling_10'].values
        features[f'{prefix}_tempo_5'] = df[f'{prefix}_T_rolling_5'].values
        features[f'{prefix}_eFG_off_5'] = df[f'{prefix}_eFG_off_rolling_5'].values
        features[f'{prefix}_eFG_def_5'] = df[f'{prefix}_eFG_def_rolling_5'].values
        features[f'{prefix}_TOV_off_5'] = df[f'{prefix}_TOV_off_rolling_5'].values
        features[f'{prefix}_TOV_def_5'] = df[f'{prefix}_TOV_def_rolling_5'].values
        features[f'{prefix}_OReb_5'] = df[f'{prefix}_OReb_rolling_5'].values
        features[f'{prefix}_DReb_5'] = df[f'{prefix}_DReb_rolling_5'].values
        features[f'{prefix}_FTR_off_5'] = df[f'{prefix}_FTR_off_rolling_5'].values
        features[f'{prefix}_FTR_def_5'] = df[f'{prefix}_FTR_def_rolling_5'].values
        features[f'{prefix}_g_score_5'] = df[f'{prefix}_g_score_rolling_5'].values
        features[f'{prefix}_sos'] = df[f'{prefix}_sos'].values
        features[f'{prefix}_def_rank'] = df[f'{prefix}_def_rank'].values
        features[f'{prefix}_off_rank'] = df[f'{prefix}_off_rank'].values
        features[f'{prefix}_barthag_rank'] = df[f'{prefix}_barthag_rank'].values
        features[f'{prefix}_3p_off'] = df[f'{prefix}_3p_off'].values
        features[f'{prefix}_3p_def'] = df[f'{prefix}_3p_def'].values
    
    # Context features
    features['is_neutral'] = (df['venue'] == 'N').astype(float).values
    features['is_conference'] = (df['home_conf'] == df['away_conf']).astype(float).values
    features['rest_diff'] = (df['home_days_rest'] - df['away_days_rest']).clip(-7, 7).values
    
    # ==========================================================================
    # LAYER 2: KNOWN GOOD FEATURES (V1's features)
    # ==========================================================================
    
    # Efficiency margins
    features['home_eff_margin_5'] = features['home_AdjO_5'] - features['home_AdjD_5']
    features['away_eff_margin_5'] = features['away_AdjO_5'] - features['away_AdjD_5']
    features['eff_margin_diff_5'] = features['home_eff_margin_5'] - features['away_eff_margin_5']
    features['eff_margin_diff_10'] = (features['home_AdjO_10'] - features['home_AdjD_10']) - \
                                      (features['away_AdjO_10'] - features['away_AdjD_10'])
    
    # Four factors diffs
    features['eFG_off_diff_5'] = features['home_eFG_off_5'] - features['away_eFG_off_5']
    features['eFG_def_diff_5'] = features['away_eFG_def_5'] - features['home_eFG_def_5']
    features['TOV_off_diff_5'] = features['away_TOV_off_5'] - features['home_TOV_off_5']
    features['TOV_def_diff_5'] = features['home_TOV_def_5'] - features['away_TOV_def_5']
    features['OReb_diff_5'] = features['home_OReb_5'] - features['away_OReb_5']
    features['DReb_diff_5'] = features['home_DReb_5'] - features['away_DReb_5']
    features['FTR_off_diff_5'] = features['home_FTR_off_5'] - features['away_FTR_off_5']
    features['FTR_def_diff_5'] = features['away_FTR_def_5'] - features['home_FTR_def_5']
    
    # Other V1 features
    features['sos_diff'] = features['home_sos'] - features['away_sos']
    features['g_score_diff_5'] = features['home_g_score_5'] - features['away_g_score_5']
    features['avg_tempo_5'] = (features['home_tempo_5'] + features['away_tempo_5']) / 2
    features['tempo_diff_5'] = features['home_tempo_5'] - features['away_tempo_5']
    
    # Home court advantage
    features['home_court'] = np.where(
        features['is_neutral'] == 1, 0.0,
        np.where(features['is_conference'] == 1, 3.0, 3.5)
    )
    
    # Rank-based features
    features['def_rank_diff'] = features['away_def_rank'] - features['home_def_rank']
    features['off_rank_diff'] = features['away_off_rank'] - features['home_off_rank']
    features['barthag_rank_diff'] = features['away_barthag_rank'] - features['home_barthag_rank']
    
    # 3pt features
    features['3p_off_diff'] = features['home_3p_off'] - features['away_3p_off']
    features['3p_def_diff'] = features['away_3p_def'] - features['home_3p_def']
    
    return features


def generate_interactions(features, max_features=500):
    """
    Generate Layer 3: Interaction features.
    Products, ratios, and other combinations.
    """
    print("\nGenerating interaction features...")
    
    new_features = {}
    
    # Key features to combine
    key_diffs = ['eff_margin_diff_5', 'eFG_off_diff_5', 'eFG_def_diff_5', 
                 'TOV_off_diff_5', 'OReb_diff_5', 'sos_diff', 'g_score_diff_5',
                 'def_rank_diff', 'barthag_rank_diff', '3p_off_diff']
    
    key_raw = ['home_eff_margin_5', 'away_eff_margin_5', 'avg_tempo_5', 
               'home_tempo_5', 'away_tempo_5', 'home_def_rank', 'away_def_rank']
    
    context = ['is_neutral', 'is_conference', 'rest_diff', 'home_court']
    
    generated = 0
    
    # Products of diffs (interaction effects)
    for f1, f2 in combinations(key_diffs, 2):
        if f1 in features and f2 in features:
            name = f'{f1}_x_{f2}'
            new_features[name] = features[f1] * features[f2]
            generated += 1
            if generated >= max_features:
                break
    
    # Diffs multiplied by context
    for diff in key_diffs:
        for ctx in context:
            if diff in features and ctx in features:
                name = f'{diff}_x_{ctx}'
                new_features[name] = features[diff] * features[ctx]
                generated += 1
    
    # Ratios (where sensible)
    # Efficiency per rank (does a high-ranked team overperform their rank?)
    if 'home_eff_margin_5' in features and 'home_barthag_rank' in features:
        # Avoid division by zero
        home_rank_safe = np.maximum(features['home_barthag_rank'], 1)
        away_rank_safe = np.maximum(features['away_barthag_rank'], 1)
        
        new_features['home_eff_per_rank'] = features['home_eff_margin_5'] / home_rank_safe * 100
        new_features['away_eff_per_rank'] = features['away_eff_margin_5'] / away_rank_safe * 100
        new_features['eff_per_rank_diff'] = new_features['home_eff_per_rank'] - new_features['away_eff_per_rank']
    
    # Tempo-adjusted features
    if 'avg_tempo_5' in features:
        tempo_factor = features['avg_tempo_5'] / 70  # normalize around typical tempo
        for diff in ['eff_margin_diff_5', 'eFG_off_diff_5']:
            if diff in features:
                new_features[f'{diff}_tempo_adj'] = features[diff] * tempo_factor
    
    # Squared terms (for non-linear effects)
    for f in ['eff_margin_diff_5', 'barthag_rank_diff', 'def_rank_diff']:
        if f in features:
            new_features[f'{f}_sq'] = features[f] ** 2
            # Also signed square (preserves direction)
            new_features[f'{f}_signed_sq'] = np.sign(features[f]) * (features[f] ** 2)
    
    # Matchup-specific: offense vs defense
    if all(f in features for f in ['home_eFG_off_5', 'away_eFG_def_5', 'away_eFG_off_5', 'home_eFG_def_5']):
        new_features['home_eFG_matchup'] = features['home_eFG_off_5'] - features['away_eFG_def_5']
        new_features['away_eFG_matchup'] = features['away_eFG_off_5'] - features['home_eFG_def_5']
        new_features['eFG_matchup_diff'] = new_features['home_eFG_matchup'] - new_features['away_eFG_matchup']
    
    # 3pt matchup
    if all(f in features for f in ['home_3p_off', 'away_3p_def', 'away_3p_off', 'home_3p_def']):
        # Ratio: how much better/worse will they shoot than usual?
        home_3p_safe = np.maximum(features['home_3p_off'], 0.01)
        away_3p_safe = np.maximum(features['away_3p_off'], 0.01)
        new_features['home_3p_matchup_ratio'] = features['away_3p_def'] / home_3p_safe
        new_features['away_3p_matchup_ratio'] = features['home_3p_def'] / away_3p_safe
        new_features['3p_matchup_ratio_diff'] = new_features['away_3p_matchup_ratio'] - new_features['home_3p_matchup_ratio']
    
    # Composite scores
    if all(f in features for f in ['eff_margin_diff_5', 'def_rank_diff', 'sos_diff']):
        # Weighted composite
        new_features['composite_v1'] = (
            features['eff_margin_diff_5'] * 0.5 +
            features['def_rank_diff'] * 0.01 +  # scale down rank
            features['sos_diff'] * 0.3
        )
    
    print(f"  Generated {len(new_features)} interaction features")
    
    return new_features


def filter_features(features, target, min_corr=0.02, max_corr_between=0.95):
    """
    Filter features:
    1. Remove those with low correlation to target
    2. Remove highly correlated pairs (keep higher corr with target)
    """
    print("\nFiltering features...")
    
    # Calculate correlations with target
    correlations = {}
    for name, values in features.items():
        # Handle NaN/Inf
        valid = np.isfinite(values) & np.isfinite(target)
        if valid.sum() < 100:
            continue
        r, _ = stats.pearsonr(values[valid], target[valid])
        if np.isfinite(r) and abs(r) >= min_corr:
            correlations[name] = abs(r)
    
    print(f"  Features with |r| >= {min_corr}: {len(correlations)}")
    
    # Sort by correlation
    sorted_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)
    
    # Remove highly correlated pairs
    kept = []
    kept_values = {}
    
    for name in sorted_features:
        values = features[name]
        
        # Check correlation with already kept features
        dominated = False
        for kept_name in kept:
            valid = np.isfinite(values) & np.isfinite(kept_values[kept_name])
            if valid.sum() < 100:
                continue
            r, _ = stats.pearsonr(values[valid], kept_values[kept_name][valid])
            if abs(r) > max_corr_between:
                dominated = True
                break
        
        if not dominated:
            kept.append(name)
            kept_values[name] = values
    
    print(f"  After removing redundant: {len(kept)}")
    
    # Return filtered features with correlations
    return {name: features[name] for name in kept}, {name: correlations[name] for name in kept}


def forward_selection(X, y, feature_names, max_features=30, cv=5):
    """
    Forward selection: greedily add features that improve CV score.
    """
    print(f"\nForward selection (max {max_features} features)...")
    
    selected = []
    remaining = list(range(len(feature_names)))
    best_score = -np.inf
    
    for i in range(max_features):
        best_candidate = None
        best_candidate_score = -np.inf
        
        for idx in remaining:
            candidate_features = selected + [idx]
            X_subset = X[:, candidate_features]
            
            # Quick evaluation with Ridge
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
            try:
                scores = cross_val_score(model, X_subset, y, cv=cv, scoring='neg_mean_absolute_error')
                score = scores.mean()
            except:
                continue
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = idx
        
        if best_candidate is None or best_candidate_score <= best_score:
            print(f"  Stopping at {len(selected)} features (no improvement)")
            break
        
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score
        
        print(f"  {i+1}. Added '{feature_names[best_candidate]}' (CV MAE: {-best_score:.3f})")
    
    return selected


def evaluate_with_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """
    Final evaluation with XGBoost.
    """
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    # Win prediction accuracy
    train_acc = np.mean((train_pred > 0) == (y_train > 0))
    test_acc = np.mean((test_pred > 0) == (y_test > 0))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'importance': importance,
        'model': model
    }


def main():
    print("=" * 70)
    print("SYSTEMATIC FEATURE DISCOVERY")
    print("=" * 70)
    
    # Load data
    df = load_data()
    target = df['margin'].values
    
    # Create base features
    print("\nCreating base features (Layer 1 & 2)...")
    features = create_base_features(df)
    print(f"  Base features: {len(features)}")
    
    # Generate interactions
    interactions = generate_interactions(features)
    features.update(interactions)
    print(f"  Total features: {len(features)}")
    
    # Filter features
    filtered_features, correlations = filter_features(features, target)
    
    # Show top correlations
    print("\nTop 20 features by correlation with margin:")
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for i, (name, corr) in enumerate(sorted_corr[:20]):
        print(f"  {i+1:2}. {name:<40} r = {corr:.4f}")
    
    # Prepare data for selection
    feature_names = list(filtered_features.keys())
    X = np.column_stack([filtered_features[name] for name in feature_names])
    
    # Handle NaN/Inf
    valid_rows = np.all(np.isfinite(X), axis=1) & np.isfinite(target)
    X = X[valid_rows]
    y = target[valid_rows]
    
    print(f"\nValid samples: {len(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Forward selection
    selected_indices = forward_selection(X_train, y_train, feature_names, max_features=25)
    selected_names = [feature_names[i] for i in selected_indices]
    
    print("\n" + "=" * 70)
    print("FINAL SELECTED FEATURES")
    print("=" * 70)
    for i, name in enumerate(selected_names):
        print(f"  {i+1:2}. {name}")
    
    # Evaluate with XGBoost
    print("\n" + "=" * 70)
    print("XGBOOST EVALUATION")
    print("=" * 70)
    
    # V1 baseline features
    v1_features = [
        'eff_margin_diff_5', 'eff_margin_diff_10',
        'eFG_off_diff_5', 'eFG_def_diff_5',
        'TOV_off_diff_5', 'TOV_def_diff_5',
        'OReb_diff_5', 'DReb_diff_5',
        'FTR_off_diff_5', 'FTR_def_diff_5',
        'home_court', 'is_neutral', 'is_conference',
        'rest_diff', 'sos_diff', 'g_score_diff_5'
    ]
    
    # Get V1 feature indices (those that exist in our filtered set)
    v1_indices = [feature_names.index(f) for f in v1_features if f in feature_names]
    
    print("\n--- V1 Baseline ---")
    v1_results = evaluate_with_xgboost(
        X_train[:, v1_indices], y_train,
        X_test[:, v1_indices], y_test,
        [feature_names[i] for i in v1_indices]
    )
    print(f"  Train MAE: {v1_results['train_mae']:.2f}")
    print(f"  Test MAE:  {v1_results['test_mae']:.2f}")
    print(f"  Train Acc: {v1_results['train_acc']:.1%}")
    print(f"  Test Acc:  {v1_results['test_acc']:.1%}")
    
    print("\n--- Selected Features ---")
    selected_results = evaluate_with_xgboost(
        X_train[:, selected_indices], y_train,
        X_test[:, selected_indices], y_test,
        selected_names
    )
    print(f"  Train MAE: {selected_results['train_mae']:.2f}")
    print(f"  Test MAE:  {selected_results['test_mae']:.2f}")
    print(f"  Train Acc: {selected_results['train_acc']:.1%}")
    print(f"  Test Acc:  {selected_results['test_acc']:.1%}")
    
    print("\n--- Top 10 Feature Importances (Selected) ---")
    for _, row in selected_results['importance'].head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mae_diff = selected_results['test_mae'] - v1_results['test_mae']
    acc_diff = selected_results['test_acc'] - v1_results['test_acc']
    print(f"MAE improvement: {mae_diff:+.2f} ({'better' if mae_diff < 0 else 'worse'})")
    print(f"Accuracy improvement: {acc_diff:+.1%} ({'better' if acc_diff > 0 else 'worse'})")
    
    if mae_diff < -0.1 and acc_diff > 0.005:
        print("\n✅ FOUND IMPROVEMENT! Consider using selected features.")
    elif mae_diff > 0.1 or acc_diff < -0.005:
        print("\n❌ Selected features worse than V1 baseline.")
    else:
        print("\n➖ Results similar to V1 baseline.")
    
    # Save selected features for reference
    print("\n" + "=" * 70)
    print("SELECTED FEATURES FOR V4")
    print("=" * 70)
    print(f"selected_features = {selected_names}")


if __name__ == '__main__':
    main()