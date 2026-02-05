"""
NCAAM Rank Analysis: Testing Barttorvik rank as a predictor

This script:
1. Tests raw rank difference as a predictor of margin
2. Tests rank bands (top 20 vs 50+, etc.)
3. Analyzes prediction accuracy by rank matchup type
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import argparse


def load_data(team_data_path, game_data_path):
    """Load team and game data."""
    teams = pd.read_csv(team_data_path)
    games = pd.read_csv(game_data_path)
    
    # Team ID in Barttorvik is their rank
    # Create rank lookup by team code
    rank_lookup = {}
    for _, row in teams.iterrows():
        code = row.get('Team_Code')
        rank = row.get('Team ID')  # This is the Barttorvik rank
        if pd.notna(code) and pd.notna(rank):
            rank_lookup[code] = int(rank)
    
    return teams, games, rank_lookup


def add_ranks_to_games(games, rank_lookup):
    """Add rank columns to game data."""
    games = games.copy()
    
    games['away_rank'] = games['away_team'].map(rank_lookup)
    games['home_rank'] = games['home_team'].map(rank_lookup)
    
    # Rank difference (negative = away team is better ranked)
    games['rank_diff'] = games['home_rank'] - games['away_rank']
    
    # Absolute rank difference
    games['rank_diff_abs'] = games['rank_diff'].abs()
    
    # Better ranked team
    games['better_ranked_team'] = np.where(
        games['away_rank'] < games['home_rank'],
        games['away_team'],
        games['home_team']
    )
    games['better_rank'] = games[['away_rank', 'home_rank']].min(axis=1)
    games['worse_rank'] = games[['away_rank', 'home_rank']].max(axis=1)
    
    return games


def assign_rank_band(rank):
    """Assign a team to a rank band."""
    if pd.isna(rank):
        return None
    rank = int(rank)
    if rank <= 20:
        return 'Elite (1-20)'
    elif rank <= 50:
        return 'Very Good (21-50)'
    elif rank <= 100:
        return 'Good (51-100)'
    elif rank <= 200:
        return 'Average (101-200)'
    else:
        return 'Below Avg (201+)'


def assign_rank_tier(rank):
    """Assign numeric tier for easier comparison."""
    if pd.isna(rank):
        return None
    rank = int(rank)
    if rank <= 20:
        return 1
    elif rank <= 50:
        return 2
    elif rank <= 100:
        return 3
    elif rank <= 200:
        return 4
    else:
        return 5


def analyze_rank_correlations(games):
    """Test correlations between rank features and outcomes."""
    print("=" * 60)
    print("RANK CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Calculate margin (home - away, positive = home win)
    games = games.copy()
    games['margin'] = games['home_score'] - games['away_score']
    games['total'] = games['home_score'] + games['away_score']
    games['home_win'] = (games['margin'] > 0).astype(int)
    
    # Filter to games with rank data
    valid = games.dropna(subset=['away_rank', 'home_rank', 'margin'])
    print(f"\nGames with rank data: {len(valid)}")
    
    # Correlations
    features = ['rank_diff', 'rank_diff_abs', 'better_rank', 'worse_rank']
    targets = ['margin', 'total', 'home_win']
    
    print("\nCorrelations (rank_diff: positive = home ranked worse):")
    print("-" * 50)
    
    for target in targets:
        print(f"\n{target.upper()}:")
        for feat in features:
            if feat in valid.columns and target in valid.columns:
                subset = valid.dropna(subset=[feat, target])
                if len(subset) > 10:
                    r, p = pearsonr(subset[feat], subset[target])
                    print(f"  {feat}: r={r:.3f} (p={p:.4f})")
    
    return valid


def analyze_rank_bands(games):
    """Analyze outcomes by rank band matchups."""
    print("\n" + "=" * 60)
    print("RANK BAND ANALYSIS")
    print("=" * 60)
    
    games = games.copy()
    games['margin'] = games['home_score'] - games['away_score']
    
    # Assign bands
    games['away_band'] = games['away_rank'].apply(assign_rank_band)
    games['home_band'] = games['home_rank'].apply(assign_rank_band)
    games['better_band'] = games['better_rank'].apply(assign_rank_band)
    games['worse_band'] = games['worse_rank'].apply(assign_rank_band)
    
    # Assign tiers
    games['away_tier'] = games['away_rank'].apply(assign_rank_tier)
    games['home_tier'] = games['home_rank'].apply(assign_rank_tier)
    games['better_tier'] = games['better_rank'].apply(assign_rank_tier)
    games['worse_tier'] = games['worse_rank'].apply(assign_rank_tier)
    games['tier_diff'] = games['worse_tier'] - games['better_tier']
    
    # Better ranked team won?
    games['better_won'] = np.where(
        games['away_rank'] < games['home_rank'],
        games['away_score'] > games['home_score'],
        games['home_score'] > games['away_score']
    )
    
    # Margin for better ranked team
    games['better_margin'] = np.where(
        games['away_rank'] < games['home_rank'],
        games['away_score'] - games['home_score'],
        games['home_score'] - games['away_score']
    )
    
    valid = games.dropna(subset=['better_tier', 'worse_tier'])
    
    # Overall: better ranked team win rate
    print(f"\nOverall better-ranked team win rate: {valid['better_won'].mean():.1%}")
    print(f"Average margin for better-ranked team: {valid['better_margin'].mean():.1f}")
    
    # By tier difference
    print("\n" + "-" * 60)
    print("BY TIER DIFFERENCE (how many tiers apart)")
    print("-" * 60)
    print(f"{'Tier Diff':<12} {'Games':>8} {'Better Wins':>12} {'Avg Margin':>12}")
    print("-" * 50)
    
    for diff in sorted(valid['tier_diff'].unique()):
        if pd.notna(diff):
            subset = valid[valid['tier_diff'] == diff]
            if len(subset) >= 5:
                win_rate = subset['better_won'].mean()
                avg_margin = subset['better_margin'].mean()
                print(f"{int(diff):<12} {len(subset):>8} {win_rate:>12.1%} {avg_margin:>12.1f}")
    
    # By better team's band
    print("\n" + "-" * 60)
    print("BY BETTER TEAM'S BAND")
    print("-" * 60)
    print(f"{'Better Band':<20} {'Games':>8} {'Win Rate':>12} {'Avg Margin':>12}")
    print("-" * 55)
    
    band_order = ['Elite (1-20)', 'Very Good (21-50)', 'Good (51-100)', 'Average (101-200)', 'Below Avg (201+)']
    for band in band_order:
        subset = valid[valid['better_band'] == band]
        if len(subset) >= 5:
            win_rate = subset['better_won'].mean()
            avg_margin = subset['better_margin'].mean()
            print(f"{band:<20} {len(subset):>8} {win_rate:>12.1%} {avg_margin:>12.1f}")
    
    # Elite vs specific bands
    print("\n" + "-" * 60)
    print("ELITE (1-20) vs OTHER BANDS")
    print("-" * 60)
    print(f"{'Opponent Band':<20} {'Games':>8} {'Elite Wins':>12} {'Avg Margin':>12}")
    print("-" * 55)
    
    elite_games = valid[valid['better_band'] == 'Elite (1-20)']
    for band in band_order[1:]:  # Skip elite vs elite
        subset = elite_games[elite_games['worse_band'] == band]
        if len(subset) >= 3:
            win_rate = subset['better_won'].mean()
            avg_margin = subset['better_margin'].mean()
            print(f"{band:<20} {len(subset):>8} {win_rate:>12.1%} {avg_margin:>12.1f}")
    
    return games


def analyze_specific_matchups(games):
    """Analyze specific matchup types for betting insights."""
    print("\n" + "=" * 60)
    print("SPECIFIC MATCHUP ANALYSIS (Betting Insights)")
    print("=" * 60)
    
    games = games.copy()
    games['margin'] = games['home_score'] - games['away_score']
    games['better_margin'] = np.where(
        games['away_rank'] < games['home_rank'],
        games['away_score'] - games['home_score'],
        games['home_score'] - games['away_score']
    )
    games['better_won'] = np.where(
        games['away_rank'] < games['home_rank'],
        games['away_score'] > games['home_score'],
        games['home_score'] > games['away_score']
    )
    
    valid = games.dropna(subset=['better_rank', 'worse_rank', 'margin'])
    
    # Define matchup types
    matchup_types = [
        ("Top 10 vs 50+", lambda g: (g['better_rank'] <= 10) & (g['worse_rank'] > 50)),
        ("Top 10 vs 100+", lambda g: (g['better_rank'] <= 10) & (g['worse_rank'] > 100)),
        ("Top 20 vs 50+", lambda g: (g['better_rank'] <= 20) & (g['worse_rank'] > 50)),
        ("Top 20 vs 100+", lambda g: (g['better_rank'] <= 20) & (g['worse_rank'] > 100)),
        ("Top 50 vs 100+", lambda g: (g['better_rank'] <= 50) & (g['worse_rank'] > 100)),
        ("Top 50 vs 200+", lambda g: (g['better_rank'] <= 50) & (g['worse_rank'] > 200)),
        ("Close matchup (within 10 ranks)", lambda g: g['rank_diff_abs'] <= 10),
        ("Close matchup (within 20 ranks)", lambda g: g['rank_diff_abs'] <= 20),
        ("Mismatch (50+ rank diff)", lambda g: g['rank_diff_abs'] >= 50),
        ("Mismatch (100+ rank diff)", lambda g: g['rank_diff_abs'] >= 100),
    ]
    
    print(f"\n{'Matchup Type':<35} {'Games':>8} {'Better Wins':>12} {'Avg Margin':>12}")
    print("-" * 70)
    
    for name, condition in matchup_types:
        subset = valid[condition(valid)]
        if len(subset) >= 3:
            win_rate = subset['better_won'].mean()
            avg_margin = subset['better_margin'].mean()
            print(f"{name:<35} {len(subset):>8} {win_rate:>12.1%} {avg_margin:>12.1f}")


def create_rank_features(games):
    """Create rank-based features for model training."""
    print("\n" + "=" * 60)
    print("RECOMMENDED RANK FEATURES FOR MODEL")
    print("=" * 60)
    
    features = [
        ("rank_diff", "Home rank - Away rank (positive = home is worse)"),
        ("rank_diff_abs", "Absolute rank difference"),
        ("better_rank", "Better team's rank (lower = better)"),
        ("worse_rank", "Worse team's rank"),
        ("tier_diff", "Tier difference (0-4 scale)"),
        ("is_mismatch_50", "1 if rank diff >= 50"),
        ("is_mismatch_100", "1 if rank diff >= 100"),
        ("is_elite_vs_avg", "1 if top 20 vs 100+"),
        ("log_rank_ratio", "log(worse_rank / better_rank)"),
    ]
    
    print("\nSuggested features to add to model:")
    for feat, desc in features:
        print(f"  • {feat}: {desc}")
    
    print("\nThese can be computed from the Team ID column in ncaam_team_data_final.csv")


def main():
    parser = argparse.ArgumentParser(description='NCAAM Rank Analysis')
    parser.add_argument('--team-data', required=True, help='Path to ncaam_team_data_final.csv')
    parser.add_argument('--game-data', required=True, help='Path to base_model_game_data_with_rolling.csv')
    
    args = parser.parse_args()
    
    print("Loading data...")
    teams, games, rank_lookup = load_data(args.team_data, args.game_data)
    print(f"  Teams with ranks: {len(rank_lookup)}")
    print(f"  Games: {len(games)}")
    
    # Add ranks to games
    games = add_ranks_to_games(games, rank_lookup)
    valid_games = games.dropna(subset=['away_rank', 'home_rank'])
    print(f"  Games with both teams ranked: {len(valid_games)}")
    
    # Run analyses
    analyze_rank_correlations(games)
    analyze_rank_bands(games)
    analyze_specific_matchups(games)
    create_rank_features(games)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()