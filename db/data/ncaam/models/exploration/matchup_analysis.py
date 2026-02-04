"""
Matchup Interaction Analysis for NCAAM

Analyzes how team strengths vs opponent weaknesses predict outcomes.
Key interactions:
1. TO Battle: Your ball security vs their pressure (and vice versa)
2. eFG Mismatch: Your shooting vs their shot defense
3. Rebound Battle: Your ORB vs their DRB
4. FT Pressure: Your FT generation vs their FT prevention

Uses rolling stats from existing game data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'processed', 'base_model_game_data_with_rolling.csv')


def load_data():
    """Load game data with rolling stats."""
    df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[''])
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to games with complete data
    required_cols = [
        'home_score', 'away_score',
        'home_eFG_off_rolling_5', 'away_eFG_off_rolling_5',
        'home_eFG_def_rolling_5', 'away_eFG_def_rolling_5',
        'home_TOV_off_rolling_5', 'away_TOV_off_rolling_5',
        'home_TOV_def_rolling_5', 'away_TOV_def_rolling_5',
        'home_OReb_rolling_5', 'away_OReb_rolling_5',
        'home_DReb_rolling_5', 'away_DReb_rolling_5',
        'home_FTR_off_rolling_5', 'away_FTR_off_rolling_5',
        'home_FTR_def_rolling_5', 'away_FTR_def_rolling_5',
    ]
    
    for col in required_cols:
        df = df[df[col].notna()]
    
    return df


def create_matchup_features(df):
    """
    Create matchup interaction features.
    
    Convention: Positive values favor home team.
    """
    features = pd.DataFrame(index=df.index)
    
    # Target
    features['margin'] = df['home_score'] - df['away_score']
    features['home_win'] = (features['margin'] > 0).astype(int)
    features['total'] = df['home_score'] + df['away_score']
    
    # ==========================================================================
    # 1. eFG MISMATCH (Shooting vs Shot Defense)
    # ==========================================================================
    # Home offense vs Away defense
    features['home_eFG_vs_away_def'] = df['home_eFG_off_rolling_5'] - df['away_eFG_def_rolling_5']
    # Away offense vs Home defense  
    features['away_eFG_vs_home_def'] = df['away_eFG_off_rolling_5'] - df['home_eFG_def_rolling_5']
    # Net mismatch (positive = home has shooting advantage)
    features['eFG_mismatch'] = features['home_eFG_vs_away_def'] - features['away_eFG_vs_home_def']
    
    # ==========================================================================
    # 2. TURNOVER BATTLE (Ball Security vs Pressure)
    # ==========================================================================
    # Note: Lower TOV% is better for offense, higher TOV_def% is better for defense
    # Home ball security vs Away pressure
    # If home has low TOV_off and away has low TOV_def (can't force TOs), home wins TO battle
    features['home_TO_battle'] = df['away_TOV_def_rolling_5'] - df['home_TOV_off_rolling_5']
    # Away ball security vs Home pressure
    features['away_TO_battle'] = df['home_TOV_def_rolling_5'] - df['away_TOV_off_rolling_5']
    # Net TO battle (positive = home wins the TO battle)
    features['TO_mismatch'] = features['home_TO_battle'] - features['away_TO_battle']
    
    # ==========================================================================
    # 3. REBOUND BATTLE (Offensive Boards vs Defensive Boards)
    # ==========================================================================
    # Home ORB vs Away DRB
    features['home_reb_battle'] = df['home_OReb_rolling_5'] - df['away_DReb_rolling_5']
    # Away ORB vs Home DRB
    features['away_reb_battle'] = df['away_OReb_rolling_5'] - df['home_DReb_rolling_5']
    # Net rebound battle (positive = home wins board battle)
    features['reb_mismatch'] = features['home_reb_battle'] - features['away_reb_battle']
    
    # ==========================================================================
    # 4. FREE THROW PRESSURE (FT Generation vs FT Prevention)
    # ==========================================================================
    # Home FT generation vs Away FT prevention
    features['home_FT_battle'] = df['home_FTR_off_rolling_5'] - df['away_FTR_def_rolling_5']
    # Away FT generation vs Home FT prevention
    features['away_FT_battle'] = df['away_FTR_off_rolling_5'] - df['home_FTR_def_rolling_5']
    # Net FT battle (positive = home gets to line more)
    features['FT_mismatch'] = features['home_FT_battle'] - features['away_FT_battle']
    
    # ==========================================================================
    # 5. COMBINED/COMPOSITE FEATURES
    # ==========================================================================
    # Possession advantage (TOs + Rebounds = extra possessions)
    features['possession_advantage'] = features['TO_mismatch'] + features['reb_mismatch']
    
    # Scoring efficiency advantage (eFG + FT = points per shot)
    features['scoring_advantage'] = features['eFG_mismatch'] + features['FT_mismatch']
    
    # Overall matchup score
    features['total_matchup_advantage'] = (
        features['eFG_mismatch'] + 
        features['TO_mismatch'] + 
        features['reb_mismatch'] + 
        features['FT_mismatch']
    )
    
    # ==========================================================================
    # 6. EXISTING V1 FEATURES (for comparison)
    # ==========================================================================
    features['eff_margin_diff'] = (
        (df['home_AdjO_rolling_5'] - df['home_AdjD_rolling_5']) -
        (df['away_AdjO_rolling_5'] - df['away_AdjD_rolling_5'])
    )
    
    # Simple diffs (what V1 currently uses)
    features['eFG_off_diff'] = df['home_eFG_off_rolling_5'] - df['away_eFG_off_rolling_5']
    features['TOV_off_diff'] = df['away_TOV_off_rolling_5'] - df['home_TOV_off_rolling_5']
    features['OReb_diff'] = df['home_OReb_rolling_5'] - df['away_OReb_rolling_5']
    features['FTR_off_diff'] = df['home_FTR_off_rolling_5'] - df['away_FTR_off_rolling_5']
    
    return features


def analyze_correlations(features):
    """Analyze correlations between matchup features and outcomes."""
    
    print("=" * 70)
    print("MATCHUP INTERACTION ANALYSIS")
    print("=" * 70)
    
    # Correlation with margin
    print("\n1. CORRELATION WITH MARGIN (higher = better predictor)")
    print("-" * 50)
    
    matchup_cols = [
        # New matchup interactions
        'eFG_mismatch',
        'TO_mismatch', 
        'reb_mismatch',
        'FT_mismatch',
        'possession_advantage',
        'scoring_advantage',
        'total_matchup_advantage',
        # Existing V1 features for comparison
        'eff_margin_diff',
        'eFG_off_diff',
        'TOV_off_diff',
        'OReb_diff',
        'FTR_off_diff',
    ]
    
    correlations = []
    for col in matchup_cols:
        r, p = stats.pearsonr(features[col], features['margin'])
        correlations.append({
            'feature': col,
            'correlation': r,
            'p_value': p,
            'abs_corr': abs(r)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    
    for _, row in corr_df.iterrows():
        marker = "NEW" if row['feature'] in ['eFG_mismatch', 'TO_mismatch', 'reb_mismatch', 
                                               'FT_mismatch', 'possession_advantage', 
                                               'scoring_advantage', 'total_matchup_advantage'] else "V1"
        print(f"  [{marker}] {row['feature']:<25} r = {row['correlation']:>7.4f}")
    
    return corr_df


def analyze_predictive_bins(features):
    """Analyze win rates by matchup advantage bins."""
    
    print("\n\n2. WIN RATE BY MATCHUP ADVANTAGE")
    print("-" * 50)
    
    for feature in ['eFG_mismatch', 'TO_mismatch', 'reb_mismatch', 'FT_mismatch', 'total_matchup_advantage']:
        print(f"\n{feature}:")
        
        # Create bins
        features['bin'] = pd.qcut(features[feature], q=5, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)'])
        
        for bin_label in ['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)']:
            bin_data = features[features['bin'] == bin_label]
            win_rate = bin_data['home_win'].mean() * 100
            avg_margin = bin_data['margin'].mean()
            print(f"  {bin_label}: {win_rate:5.1f}% win rate, {avg_margin:+5.1f} avg margin (n={len(bin_data)})")
        
        features.drop('bin', axis=1, inplace=True)


def analyze_incremental_value(features):
    """Check if matchup features add value beyond eff_margin_diff."""
    
    print("\n\n3. INCREMENTAL VALUE ANALYSIS")
    print("-" * 50)
    print("(Partial correlations controlling for eff_margin_diff)")
    
    from scipy.stats import pearsonr
    
    # Residualize margin on eff_margin_diff
    slope, intercept = np.polyfit(features['eff_margin_diff'], features['margin'], 1)
    residual_margin = features['margin'] - (slope * features['eff_margin_diff'] + intercept)
    
    # Check what correlates with the residual
    matchup_cols = ['eFG_mismatch', 'TO_mismatch', 'reb_mismatch', 'FT_mismatch', 
                    'possession_advantage', 'scoring_advantage', 'total_matchup_advantage']
    
    print("\nCorrelation with margin AFTER accounting for eff_margin_diff:")
    for col in matchup_cols:
        # Also residualize the feature
        slope2, intercept2 = np.polyfit(features['eff_margin_diff'], features[col], 1)
        residual_feature = features[col] - (slope2 * features['eff_margin_diff'] + intercept2)
        
        r, p = pearsonr(residual_feature, residual_margin)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<25} r = {r:>7.4f} {sig}")


def analyze_extreme_matchups(features):
    """Look at games with extreme matchup advantages."""
    
    print("\n\n4. EXTREME MATCHUP ANALYSIS")
    print("-" * 50)
    
    for feature in ['eFG_mismatch', 'TO_mismatch', 'reb_mismatch', 'total_matchup_advantage']:
        p90 = features[feature].quantile(0.90)
        p10 = features[feature].quantile(0.10)
        
        strong_home = features[features[feature] >= p90]
        strong_away = features[features[feature] <= p10]
        middle = features[(features[feature] > p10) & (features[feature] < p90)]
        
        print(f"\n{feature}:")
        print(f"  Strong home advantage (top 10%): {strong_home['home_win'].mean()*100:.1f}% home win, {strong_home['margin'].mean():+.1f} margin (n={len(strong_home)})")
        print(f"  Middle 80%:                      {middle['home_win'].mean()*100:.1f}% home win, {middle['margin'].mean():+.1f} margin (n={len(middle)})")
        print(f"  Strong away advantage (bot 10%): {strong_away['home_win'].mean()*100:.1f}% home win, {strong_away['margin'].mean():+.1f} margin (n={len(strong_away)})")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Games: {len(df)}")
    
    print("Creating matchup features...")
    features = create_matchup_features(df)
    
    corr_df = analyze_correlations(features)
    analyze_predictive_bins(features)
    analyze_incremental_value(features)
    analyze_extreme_matchups(features)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Get top new features
    new_features = corr_df[corr_df['feature'].isin([
        'eFG_mismatch', 'TO_mismatch', 'reb_mismatch', 'FT_mismatch',
        'possession_advantage', 'scoring_advantage', 'total_matchup_advantage'
    ])]
    
    print("\nTop NEW matchup features by correlation:")
    for _, row in new_features.head(5).iterrows():
        print(f"  {row['feature']:<25} r = {row['correlation']:.4f}")
    
    print("\nRecommendation: Add features with |r| > 0.10 that provide incremental value")


if __name__ == '__main__':
    main()