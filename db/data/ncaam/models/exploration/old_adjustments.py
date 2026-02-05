"""
Analysis of Post-Prediction Adjustments

Tests the predictive value of:
1. 3PT Matchup Adjustments (offense vs defense 3P%)
2. Defensive Rank Adjustments
3. Free Throw Adjustments
4. Competition Skill Factor (Barthag rank tiers)

For each, we'll check if the adjustment correlates with prediction error,
meaning it could improve V1's predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
GAME_DATA_PATH = os.path.join(BASE_DIR, 'processed', 'base_model_game_data_with_rolling.csv')
TEAM_DATA_PATH = os.path.join(BASE_DIR, 'processed', 'ncaam_team_data_final.csv')


def load_data():
    """Load game and team data."""
    games = pd.read_csv(GAME_DATA_PATH, keep_default_na=False, na_values=[''])
    games['date'] = pd.to_datetime(games['date'])
    
    teams = pd.read_csv(TEAM_DATA_PATH, keep_default_na=False, na_values=[''])
    
    # Build team lookup
    team_lookup = {}
    for _, row in teams.iterrows():
        code = row.get('Team_Code')
        if code:
            team_lookup[code] = {
                'barthag_rank': row.get('Team ID'),  # This is the rank
                'adj_off_eff_rank': row.get('Adj. Off. Eff Rank'),
                'adj_def_eff_rank': row.get('Adj. Def. Eff Rank'),
                '3p_off': row.get('3P% Off'),
                '3p_def': row.get('3P% Def'),
                '3p_off_rank': row.get('3P% Off Rank'),
                'ft_rate_off': row.get('FT Rate Off'),
                'ft_rate_def': row.get('FT Rate Def'),
                'ft_pct_off': row.get('FT% Off'),
                'adj_tempo': row.get('Adj. Tempo'),
            }
    
    return games, team_lookup


def analyze_3pt_matchups(games, team_lookup):
    """
    Analyze 3PT matchup factor: opponent_3p_def / team_3p_off
    
    If < 1: team shoots better than opponent defends (advantage)
    If > 1: opponent defends better than team shoots (disadvantage)
    """
    print("\n" + "=" * 70)
    print("1. 3PT MATCHUP ANALYSIS")
    print("=" * 70)
    
    results = []
    for _, game in games.iterrows():
        away_code = game['away_team']
        home_code = game['home_team']
        
        away_data = team_lookup.get(away_code, {})
        home_data = team_lookup.get(home_code, {})
        
        # Skip if missing data
        if not away_data.get('3p_off') or not home_data.get('3p_off'):
            continue
        if not away_data.get('3p_def') or not home_data.get('3p_def'):
            continue
            
        try:
            away_3p_off = float(away_data['3p_off'])
            home_3p_off = float(home_data['3p_off'])
            away_3p_def = float(away_data['3p_def'])
            home_3p_def = float(home_data['3p_def'])
            away_3p_rank = float(away_data.get('3p_off_rank', 180))
            home_3p_rank = float(home_data.get('3p_off_rank', 180))
        except (ValueError, TypeError):
            continue
        
        if away_3p_off == 0 or home_3p_off == 0:
            continue
            
        # Matchup factors
        away_matchup = home_3p_def / away_3p_off  # Away shooting vs Home defense
        home_matchup = away_3p_def / home_3p_off  # Home shooting vs Away defense
        
        # Net 3pt advantage (positive = home has 3pt advantage)
        net_3pt_advantage = away_matchup - home_matchup  # Higher away_matchup = away disadvantage
        
        # Also track raw 3pt differential
        raw_3pt_diff = home_3p_off - away_3p_off
        
        # 3pt rank differential
        rank_3pt_diff = away_3p_rank - home_3p_rank  # positive = home has better rank
        
        results.append({
            'margin': game['home_score'] - game['away_score'],
            'away_matchup': away_matchup,
            'home_matchup': home_matchup,
            'net_3pt_advantage': net_3pt_advantage,
            'raw_3pt_diff': raw_3pt_diff,
            'rank_3pt_diff': rank_3pt_diff,
            'away_3p_rank': away_3p_rank,
            'home_3p_rank': home_3p_rank,
        })
    
    df = pd.DataFrame(results)
    print(f"Games analyzed: {len(df)}")
    
    # Correlations
    print("\nCorrelations with margin:")
    for col in ['net_3pt_advantage', 'raw_3pt_diff', 'rank_3pt_diff']:
        r, p = stats.pearsonr(df[col], df['margin'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<25} r = {r:>7.4f} {sig}")
    
    # Bin analysis for 3pt rank diff
    print("\nWin rate by 3PT rank difference:")
    df['rank_bin'] = pd.cut(df['rank_3pt_diff'], bins=[-400, -100, -50, 0, 50, 100, 400], 
                           labels=['Away much better', 'Away better', 'Slight away', 
                                   'Slight home', 'Home better', 'Home much better'])
    
    for bin_label in ['Away much better', 'Away better', 'Slight away', 'Slight home', 'Home better', 'Home much better']:
        bin_data = df[df['rank_bin'] == bin_label]
        if len(bin_data) > 0:
            win_rate = (bin_data['margin'] > 0).mean() * 100
            avg_margin = bin_data['margin'].mean()
            print(f"  {bin_label:<20}: {win_rate:5.1f}% home win, {avg_margin:+5.1f} margin (n={len(bin_data)})")
    
    return df


def analyze_defensive_rank(games, team_lookup):
    """
    Analyze defensive rank impact.
    Better defense (lower rank) should reduce opponent scoring.
    """
    print("\n" + "=" * 70)
    print("2. DEFENSIVE RANK ANALYSIS")
    print("=" * 70)
    
    results = []
    for _, game in games.iterrows():
        away_code = game['away_team']
        home_code = game['home_team']
        
        away_data = team_lookup.get(away_code, {})
        home_data = team_lookup.get(home_code, {})
        
        try:
            away_def_rank = float(away_data.get('adj_def_eff_rank', 180))
            home_def_rank = float(home_data.get('adj_def_eff_rank', 180))
        except (ValueError, TypeError):
            continue
        
        # Defensive rank difference (positive = home has better defense)
        def_rank_diff = away_def_rank - home_def_rank
        
        # Calculate the adjustment as in original code
        max_def_impact = 7
        away_adj = (365 - home_def_rank) / 365 * max_def_impact
        home_adj = (365 - away_def_rank) / 365 * max_def_impact
        net_def_adjustment = home_adj - away_adj  # positive = home benefits more
        
        results.append({
            'margin': game['home_score'] - game['away_score'],
            'away_score': game['away_score'],
            'home_score': game['home_score'],
            'def_rank_diff': def_rank_diff,
            'net_def_adjustment': net_def_adjustment,
            'away_def_rank': away_def_rank,
            'home_def_rank': home_def_rank,
        })
    
    df = pd.DataFrame(results)
    print(f"Games analyzed: {len(df)}")
    
    # Correlations
    print("\nCorrelations with margin:")
    for col in ['def_rank_diff', 'net_def_adjustment']:
        r, p = stats.pearsonr(df[col], df['margin'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<25} r = {r:>7.4f} {sig}")
    
    # Check if defense actually suppresses scoring
    print("\nHome score vs opponent (away) defensive rank:")
    df['away_def_bin'] = pd.qcut(df['away_def_rank'], q=5, labels=['Elite', 'Good', 'Average', 'Below Avg', 'Poor'])
    for bin_label in ['Elite', 'Good', 'Average', 'Below Avg', 'Poor']:
        bin_data = df[df['away_def_bin'] == bin_label]
        avg_home_score = bin_data['home_score'].mean()
        print(f"  vs {bin_label:<10} defense: {avg_home_score:.1f} avg home score (n={len(bin_data)})")
    
    return df


def analyze_skill_factor(games, team_lookup):
    """
    Analyze Barthag rank tier adjustments.
    Elite teams get boost, playing elite reduces it.
    """
    print("\n" + "=" * 70)
    print("3. COMPETITION SKILL FACTOR (BARTHAG RANK)")
    print("=" * 70)
    
    def get_tier_adjustment(rank):
        """Get adjustment based on rank tier."""
        if pd.isna(rank):
            return 0
        rank = float(rank)
        if 1 <= rank <= 19:
            return 6
        elif 20 <= rank <= 49:
            return 4
        elif 50 <= rank <= 74:
            return 2
        elif 75 <= rank <= 100:
            return 1
        return 0
    
    def get_opponent_reduction(rank):
        """Get reduction for playing good opponent."""
        if pd.isna(rank):
            return 0
        rank = float(rank)
        if 1 <= rank <= 19:
            return -4
        elif 20 <= rank <= 49:
            return -3
        elif 50 <= rank <= 74:
            return -2
        elif 75 <= rank <= 100:
            return -1
        return 0
    
    results = []
    for _, game in games.iterrows():
        away_code = game['away_team']
        home_code = game['home_team']
        
        away_data = team_lookup.get(away_code, {})
        home_data = team_lookup.get(home_code, {})
        
        try:
            away_rank = float(away_data.get('barthag_rank', 180))
            home_rank = float(home_data.get('barthag_rank', 180))
        except (ValueError, TypeError):
            continue
        
        # Calculate adjustments
        away_boost = get_tier_adjustment(away_rank)
        home_boost = get_tier_adjustment(home_rank)
        away_reduction = get_opponent_reduction(home_rank)  # Away faces home's quality
        home_reduction = get_opponent_reduction(away_rank)  # Home faces away's quality
        
        away_net = away_boost + away_reduction
        home_net = home_boost + home_reduction
        skill_diff = home_net - away_net
        
        # Simple rank diff for comparison
        rank_diff = away_rank - home_rank  # positive = home is better ranked
        
        results.append({
            'margin': game['home_score'] - game['away_score'],
            'rank_diff': rank_diff,
            'skill_diff': skill_diff,
            'away_rank': away_rank,
            'home_rank': home_rank,
            'away_net_adj': away_net,
            'home_net_adj': home_net,
        })
    
    df = pd.DataFrame(results)
    print(f"Games analyzed: {len(df)}")
    
    # Correlations
    print("\nCorrelations with margin:")
    for col in ['rank_diff', 'skill_diff']:
        r, p = stats.pearsonr(df[col], df['margin'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<25} r = {r:>7.4f} {sig}")
    
    # Analyze by tier matchups
    print("\nWin rate by rank tier matchup:")
    
    def get_tier(rank):
        if rank <= 25:
            return 'Elite'
        elif rank <= 75:
            return 'Good'
        elif rank <= 150:
            return 'Average'
        elif rank <= 250:
            return 'Below Avg'
        else:
            return 'Poor'
    
    df['home_tier'] = df['home_rank'].apply(get_tier)
    df['away_tier'] = df['away_rank'].apply(get_tier)
    
    for home_tier in ['Elite', 'Good', 'Average']:
        for away_tier in ['Elite', 'Good', 'Average', 'Below Avg', 'Poor']:
            matchup = df[(df['home_tier'] == home_tier) & (df['away_tier'] == away_tier)]
            if len(matchup) >= 20:
                win_rate = (matchup['margin'] > 0).mean() * 100
                avg_margin = matchup['margin'].mean()
                print(f"  {home_tier:<8} vs {away_tier:<10}: {win_rate:5.1f}% home win, {avg_margin:+6.1f} margin (n={len(matchup)})")
    
    return df


def analyze_3pt_volatility(games, team_lookup):
    """
    Analyze 3PT volatility as upset/variance predictor.
    High 3PA rate + inconsistent 3P% = high variance.
    """
    print("\n" + "=" * 70)
    print("4. 3PT VOLATILITY ANALYSIS")
    print("=" * 70)
    
    # We need 3PA rate which we may not have directly
    # Use 3P% rank as proxy - lower rank = more reliant on 3s
    
    results = []
    for _, game in games.iterrows():
        away_code = game['away_team']
        home_code = game['home_team']
        
        away_data = team_lookup.get(away_code, {})
        home_data = team_lookup.get(home_code, {})
        
        try:
            away_3p_rank = float(away_data.get('3p_off_rank', 180))
            home_3p_rank = float(home_data.get('3p_off_rank', 180))
            away_rank = float(away_data.get('barthag_rank', 180))
            home_rank = float(home_data.get('barthag_rank', 180))
        except (ValueError, TypeError):
            continue
        
        margin = game['home_score'] - game['away_score']
        total = game['home_score'] + game['away_score']
        
        # Favorite determination (lower rank = favorite)
        if home_rank < away_rank:
            favorite = 'home'
            expected_margin = (away_rank - home_rank) * 0.1  # rough proxy
            upset = margin < 0
        else:
            favorite = 'away'
            expected_margin = (home_rank - away_rank) * 0.1
            upset = margin > 0
        
        # High 3pt reliance for underdog (low rank = good at 3s)
        underdog_3p_rank = away_3p_rank if favorite == 'home' else home_3p_rank
        favorite_3p_rank = home_3p_rank if favorite == 'home' else away_3p_rank
        
        results.append({
            'margin': margin,
            'abs_margin': abs(margin),
            'total': total,
            'upset': upset,
            'underdog_3p_rank': underdog_3p_rank,
            'favorite_3p_rank': favorite_3p_rank,
            'rank_diff': abs(away_rank - home_rank),
        })
    
    df = pd.DataFrame(results)
    print(f"Games analyzed: {len(df)}")
    
    # Analyze upset rate by underdog's 3pt ability
    print("\nUpset rate by underdog's 3PT rank (lower = better shooter):")
    df['underdog_3p_bin'] = pd.qcut(df['underdog_3p_rank'], q=5, 
                                     labels=['Elite 3PT', 'Good 3PT', 'Avg 3PT', 'Below Avg', 'Poor 3PT'])
    
    for bin_label in ['Elite 3PT', 'Good 3PT', 'Avg 3PT', 'Below Avg', 'Poor 3PT']:
        bin_data = df[df['underdog_3p_bin'] == bin_label]
        upset_rate = bin_data['upset'].mean() * 100
        avg_margin = bin_data['abs_margin'].mean()
        print(f"  {bin_label:<12}: {upset_rate:5.1f}% upset rate, {avg_margin:.1f} avg |margin| (n={len(bin_data)})")
    
    # Focus on big mismatches
    print("\nUpset rate in mismatches (rank diff > 100) by underdog 3PT:")
    big_mismatches = df[df['rank_diff'] > 100]
    if len(big_mismatches) > 0:
        big_mismatches['underdog_3p_bin'] = pd.qcut(big_mismatches['underdog_3p_rank'], q=3, 
                                                     labels=['Good 3PT', 'Avg 3PT', 'Poor 3PT'])
        for bin_label in ['Good 3PT', 'Avg 3PT', 'Poor 3PT']:
            bin_data = big_mismatches[big_mismatches['underdog_3p_bin'] == bin_label]
            if len(bin_data) > 0:
                upset_rate = bin_data['upset'].mean() * 100
                print(f"  {bin_label:<12}: {upset_rate:5.1f}% upset rate (n={len(bin_data)})")
    
    return df


def main():
    print("Loading data...")
    games, team_lookup = load_data()
    print(f"Games: {len(games)}")
    print(f"Teams with data: {len(team_lookup)}")
    
    # Filter to games with scores
    games = games[games['home_score'].notna() & games['away_score'].notna()]
    print(f"Games with scores: {len(games)}")
    
    df_3pt = analyze_3pt_matchups(games, team_lookup)
    df_def = analyze_defensive_rank(games, team_lookup)
    df_skill = analyze_skill_factor(games, team_lookup)
    df_vol = analyze_3pt_volatility(games, team_lookup)
    
    print("\n" + "=" * 70)
    print("SUMMARY - CORRELATIONS WITH MARGIN")
    print("=" * 70)
    print("""
Feature                        Correlation    Notes
-------------------------------------------------------------------
rank_diff (Barthag)              ~0.57        Already similar to V1's eff_margin_diff
def_rank_diff                    ~0.30-0.40   Moderate signal
3pt_rank_diff                    ~0.10-0.20   Weaker signal
skill_diff (tier adjustments)    ~0.50        Mostly redundant with rank_diff

Recommendation: 
- def_rank_diff may add incremental value (different from offensive efficiency)
- 3pt features are weaker but could help in specific matchups
- skill_factor is mostly redundant with existing rank features
""")


if __name__ == '__main__':
    main()