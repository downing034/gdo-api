"""
NCAAM Quad Analysis: Testing quad-level features as predictors

This script:
1. Calculates quad level for each game based on opponent rank and location
2. Computes team quad records (Q1 wins, Q4 losses, etc.)
3. Tests correlations between quad features and outcomes
4. Identifies which quad features are most predictive
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
import argparse


def get_quad(opp_rank, location):
    """
    Determine quad level based on opponent rank and game location.
    
    Args:
        opp_rank: Opponent's Barttorvik rank (1-363)
        location: 'home', 'away', or 'neutral'
    
    Returns:
        Quad level (1-4)
    """
    if pd.isna(opp_rank):
        return None
    
    opp_rank = int(opp_rank)
    
    if location == 'home':
        if opp_rank <= 30: return 1
        elif opp_rank <= 75: return 2
        elif opp_rank <= 160: return 3
        else: return 4
    elif location == 'neutral':
        if opp_rank <= 50: return 1
        elif opp_rank <= 100: return 2
        elif opp_rank <= 200: return 3
        else: return 4
    else:  # away
        if opp_rank <= 75: return 1
        elif opp_rank <= 135: return 2
        elif opp_rank <= 240: return 3
        else: return 4


def load_data(team_data_path, game_data_path):
    """Load team and game data."""
    teams = pd.read_csv(team_data_path, keep_default_na=False, na_values=[''])
    games = pd.read_csv(game_data_path, keep_default_na=False, na_values=[''])
    
    # Create rank lookup by team code
    rank_lookup = {}
    for _, row in teams.iterrows():
        code = row.get('Team_Code')
        rank = row.get('Team ID')
        if code and rank:
            try:
                rank_lookup[code] = int(float(rank))
            except (ValueError, TypeError):
                pass
    
    return teams, games, rank_lookup


def add_game_quads(games, rank_lookup):
    """Add quad level to each game row. Expands to two rows per game (one per team)."""
    
    expanded_rows = []
    
    for _, row in games.iterrows():
        away_team = row.get('away_team')
        home_team = row.get('home_team')
        away_score = row.get('away_score')
        home_score = row.get('home_score')
        
        away_rank = rank_lookup.get(away_team)
        home_rank = rank_lookup.get(home_team)
        
        # Skip if missing data
        if not away_team or not home_team:
            continue
        if pd.isna(away_score) or pd.isna(home_score):
            continue
            
        # Away team's perspective
        away_row = {
            'game_id': row.get('id'),
            'date': row.get('date'),
            'team': away_team,
            'opponent': home_team,
            'team_rank': away_rank,
            'opp_rank': home_rank,
            'location': 'away',
            'score': away_score,
            'opp_score': home_score,
            'win': away_score > home_score,
            'game_quad': get_quad(home_rank, 'away'),  # Away game vs home_rank opponent
        }
        expanded_rows.append(away_row)
        
        # Home team's perspective
        home_row = {
            'game_id': row.get('id'),
            'date': row.get('date'),
            'team': home_team,
            'opponent': away_team,
            'team_rank': home_rank,
            'opp_rank': away_rank,
            'location': 'home',
            'score': home_score,
            'opp_score': away_score,
            'win': home_score > away_score,
            'game_quad': get_quad(away_rank, 'home'),  # Home game vs away_rank opponent
        }
        expanded_rows.append(home_row)
    
    return pd.DataFrame(expanded_rows)


def calculate_team_quad_records(games):
    """
    Calculate quad records for each team based on their game history.
    
    Returns dict: team_code -> {q1_wins, q1_losses, q2_wins, etc.}
    """
    team_records = defaultdict(lambda: {
        'q1_wins': 0, 'q1_losses': 0,
        'q2_wins': 0, 'q2_losses': 0,
        'q3_wins': 0, 'q3_losses': 0,
        'q4_wins': 0, 'q4_losses': 0,
        'total_games': 0
    })
    
    for _, row in games.iterrows():
        team = row.get('team')
        quad = row.get('game_quad')
        win = row.get('win')
        
        if not team or pd.isna(quad) or pd.isna(win):
            continue
        
        quad = int(quad)
        record = team_records[team]
        record['total_games'] += 1
        
        if win:
            record[f'q{quad}_wins'] += 1
        else:
            record[f'q{quad}_losses'] += 1
    
    # Calculate derived metrics
    for team, record in team_records.items():
        # Quality wins (Q1 + Q2)
        record['q1_q2_wins'] = record['q1_wins'] + record['q2_wins']
        record['q1_q2_games'] = record['q1_wins'] + record['q1_losses'] + record['q2_wins'] + record['q2_losses']
        record['q1_q2_win_pct'] = (
            record['q1_q2_wins'] / record['q1_q2_games'] 
            if record['q1_q2_games'] > 0 else 0
        )
        
        # Bad losses (Q3 + Q4)
        record['q3_q4_losses'] = record['q3_losses'] + record['q4_losses']
        
        # Q1 win rate
        q1_games = record['q1_wins'] + record['q1_losses']
        record['q1_win_pct'] = record['q1_wins'] / q1_games if q1_games > 0 else 0
        
        # Overall quality score: Q1 wins - Q4 losses (simple metric)
        record['quality_score'] = record['q1_wins'] - record['q4_losses']
        
        # Weighted quality: Q1 wins worth more, Q4 losses penalized more
        record['weighted_quality'] = (
            3 * record['q1_wins'] + 
            2 * record['q2_wins'] + 
            1 * record['q3_wins'] +
            0 * record['q4_wins'] -
            1 * record['q3_losses'] -
            3 * record['q4_losses']
        )
    
    return dict(team_records)


def analyze_quad_correlations(games, team_records, rank_lookup):
    """Test correlations between quad features and game outcomes."""
    print("=" * 70)
    print("QUAD FEATURE CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Build dataset with quad features for each game
    rows = []
    for _, row in games.iterrows():
        team = row.get('team')
        opp = row.get('opponent')
        
        if team not in team_records or opp not in team_records:
            continue
        
        team_rec = team_records[team]
        opp_rec = team_records[opp]
        
        # Skip if not enough games
        if team_rec['total_games'] < 5 or opp_rec['total_games'] < 5:
            continue
        
        game_row = {
            # Targets
            'win': 1 if row.get('win') else 0,
            'margin': row.get('score', 0) - row.get('opp_score', 0) if 'score' in row else row.get('target_score', 0) - row.get('target_opponent_score', 0),
            
            # Game context
            'game_quad': row.get('game_quad'),
            'team_rank': rank_lookup.get(team),
            'opp_rank': rank_lookup.get(opp),
            
            # Team quad features
            'team_q1_wins': team_rec['q1_wins'],
            'team_q1_losses': team_rec['q1_losses'],
            'team_q2_wins': team_rec['q2_wins'],
            'team_q1_q2_wins': team_rec['q1_q2_wins'],
            'team_q1_q2_win_pct': team_rec['q1_q2_win_pct'],
            'team_q3_q4_losses': team_rec['q3_q4_losses'],
            'team_q4_losses': team_rec['q4_losses'],
            'team_quality_score': team_rec['quality_score'],
            'team_weighted_quality': team_rec['weighted_quality'],
            
            # Opponent quad features
            'opp_q1_wins': opp_rec['q1_wins'],
            'opp_q1_losses': opp_rec['q1_losses'],
            'opp_q2_wins': opp_rec['q2_wins'],
            'opp_q1_q2_wins': opp_rec['q1_q2_wins'],
            'opp_q1_q2_win_pct': opp_rec['q1_q2_win_pct'],
            'opp_q3_q4_losses': opp_rec['q3_q4_losses'],
            'opp_q4_losses': opp_rec['q4_losses'],
            'opp_quality_score': opp_rec['quality_score'],
            'opp_weighted_quality': opp_rec['weighted_quality'],
            
            # Differential features
            'q1_wins_diff': team_rec['q1_wins'] - opp_rec['q1_wins'],
            'q1_q2_wins_diff': team_rec['q1_q2_wins'] - opp_rec['q1_q2_wins'],
            'q1_q2_win_pct_diff': team_rec['q1_q2_win_pct'] - opp_rec['q1_q2_win_pct'],
            'q3_q4_losses_diff': team_rec['q3_q4_losses'] - opp_rec['q3_q4_losses'],
            'q4_losses_diff': team_rec['q4_losses'] - opp_rec['q4_losses'],
            'quality_score_diff': team_rec['quality_score'] - opp_rec['quality_score'],
            'weighted_quality_diff': team_rec['weighted_quality'] - opp_rec['weighted_quality'],
        }
        rows.append(game_row)
    
    df = pd.DataFrame(rows)
    print(f"\nGames with quad data: {len(df)}")
    
    # Correlations with margin
    print("\n" + "-" * 70)
    print("CORRELATIONS WITH MARGIN")
    print("-" * 70)
    
    features = [
        'team_q1_wins', 'team_q1_q2_wins', 'team_q1_q2_win_pct', 
        'team_q3_q4_losses', 'team_q4_losses',
        'team_quality_score', 'team_weighted_quality',
        'opp_q1_wins', 'opp_q1_q2_wins', 'opp_q3_q4_losses',
        'q1_wins_diff', 'q1_q2_wins_diff', 'q1_q2_win_pct_diff',
        'q3_q4_losses_diff', 'q4_losses_diff',
        'quality_score_diff', 'weighted_quality_diff',
        'team_rank', 'opp_rank',  # For comparison
    ]
    
    correlations = []
    for feat in features:
        if feat in df.columns:
            valid = df.dropna(subset=[feat, 'margin'])
            if len(valid) > 10:
                r, p = pearsonr(valid[feat], valid['margin'])
                correlations.append((feat, r, p, len(valid)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{'Feature':<30} {'r':>10} {'p-value':>12} {'n':>8}")
    print("-" * 65)
    for feat, r, p, n in correlations:
        print(f"{feat:<30} {r:>10.3f} {p:>12.4f} {n:>8}")
    
    # Correlations with win
    print("\n" + "-" * 70)
    print("CORRELATIONS WITH WIN")
    print("-" * 70)
    
    correlations = []
    for feat in features:
        if feat in df.columns:
            valid = df.dropna(subset=[feat, 'win'])
            if len(valid) > 10:
                r, p = pearsonr(valid[feat], valid['win'])
                correlations.append((feat, r, p, len(valid)))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{'Feature':<30} {'r':>10} {'p-value':>12} {'n':>8}")
    print("-" * 65)
    for feat, r, p, n in correlations:
        print(f"{feat:<30} {r:>10.3f} {p:>12.4f} {n:>8}")
    
    return df


def analyze_by_game_quad(games, team_records, rank_lookup):
    """Analyze outcomes by the quad level of the game itself."""
    print("\n" + "=" * 70)
    print("OUTCOMES BY GAME QUAD LEVEL")
    print("=" * 70)
    
    valid = games.dropna(subset=['game_quad', 'win'])
    
    print(f"\n{'Quad':<8} {'Games':>10} {'Wins':>10} {'Win Rate':>12} {'Avg Margin':>12}")
    print("-" * 55)
    
    for quad in [1, 2, 3, 4]:
        subset = valid[valid['game_quad'] == quad]
        if len(subset) > 0:
            wins = subset['win'].sum()
            win_rate = wins / len(subset)
            avg_margin = (subset['score'] - subset['opp_score']).mean()
            
            print(f"Q{quad:<7} {len(subset):>10} {int(wins):>10} {win_rate:>12.1%} {avg_margin:>12.1f}")
    
    # Better ranked team performance by quad
    print("\n" + "-" * 70)
    print("BETTER-RANKED TEAM WIN RATE BY GAME QUAD")
    print("-" * 70)
    
    valid_with_ranks = valid.dropna(subset=['team_rank', 'opp_rank'])
    valid_with_ranks = valid_with_ranks.copy()
    valid_with_ranks['is_better_ranked'] = valid_with_ranks['team_rank'] < valid_with_ranks['opp_rank']
    valid_with_ranks['better_won'] = (
        (valid_with_ranks['is_better_ranked'] & valid_with_ranks['win']) | 
        (~valid_with_ranks['is_better_ranked'] & ~valid_with_ranks['win'])
    )
    
    print(f"\n{'Quad':<8} {'Games':>10} {'Better Wins':>14} {'Win Rate':>12}")
    print("-" * 50)
    
    for quad in [1, 2, 3, 4]:
        subset = valid_with_ranks[valid_with_ranks['game_quad'] == quad]
        if len(subset) > 0:
            better_wins = subset['better_won'].sum()
            win_rate = better_wins / len(subset)
            print(f"Q{quad:<7} {len(subset):>10} {int(better_wins):>14} {win_rate:>12.1%}")


def analyze_q1_vs_q4_teams(games, team_records, rank_lookup):
    """Compare teams with strong Q1 records vs teams with Q4 losses."""
    print("\n" + "=" * 70)
    print("Q1 WINNERS VS Q4 LOSERS ANALYSIS")
    print("=" * 70)
    
    # Categorize teams
    q1_strong = []  # 3+ Q1 wins
    q4_weak = []    # 2+ Q4 losses
    
    for team, rec in team_records.items():
        if rec['q1_wins'] >= 3:
            q1_strong.append(team)
        if rec['q4_losses'] >= 2:
            q4_weak.append(team)
    
    print(f"\nTeams with 3+ Q1 wins: {len(q1_strong)}")
    print(f"Teams with 2+ Q4 losses: {len(q4_weak)}")
    
    # Find matchups between these groups
    games = games.copy()
    
    def is_q1_strong(team):
        return team in q1_strong
    
    def is_q4_weak(team):
        return team in q4_weak
    
    games['team_q1_strong'] = games['team'].apply(is_q1_strong)
    games['opp_q1_strong'] = games['opponent'].apply(is_q1_strong)
    games['team_q4_weak'] = games['team'].apply(is_q4_weak)
    games['opp_q4_weak'] = games['opponent'].apply(is_q4_weak)
    
    # Q1 strong vs Q4 weak matchups
    matchups = games[
        (games['team_q1_strong'] & games['opp_q4_weak']) |
        (games['team_q4_weak'] & games['opp_q1_strong'])
    ].dropna(subset=['win'])
    
    if len(matchups) > 0:
        # Normalize: team is always the Q1 strong team
        def q1_strong_won(row):
            if row['team_q1_strong']:
                return row['win']
            else:
                return not row['win']
        
        matchups['q1_strong_won'] = matchups.apply(q1_strong_won, axis=1)
        
        wins = matchups['q1_strong_won'].sum()
        total = len(matchups)
        
        print(f"\nQ1-Strong vs Q4-Weak matchups: {total}")
        print(f"Q1-Strong team won: {int(wins)}/{total} ({100*wins/total:.1f}%)")


def recommend_features(correlations_df):
    """Recommend which quad features to add to the model."""
    print("\n" + "=" * 70)
    print("RECOMMENDED QUAD FEATURES FOR MODEL")
    print("=" * 70)
    
    print("""
Based on the correlation analysis, consider adding these features:

DIFFERENTIAL FEATURES (most predictive):
  • quality_score_diff: Team Q1 wins minus Q4 losses, differenced
  • weighted_quality_diff: Weighted sum of quad wins/losses, differenced
  • q1_q2_win_pct_diff: Quality win percentage difference

TEAM-SPECIFIC FEATURES:
  • team_q1_wins: Number of Q1 wins (battle-tested indicator)
  • team_q4_losses: Number of Q4 losses (red flag indicator)
  • team_q1_q2_win_pct: Win rate in quality games

GAME CONTEXT:
  • game_quad: The quad level of THIS specific matchup (1-4)
  • is_q1_game: Binary flag for Q1 games (high-stakes indicator)

COMPUTATION NOTES:
  - Quad features require tracking game history per team
  - Will need to compute in data processor, not at prediction time
  - Early season will have sparse data (consider minimum games threshold)
""")


def main():
    parser = argparse.ArgumentParser(description='NCAAM Quad Analysis')
    parser.add_argument('--team-data', required=True, help='Path to ncaam_team_data_final.csv')
    parser.add_argument('--game-data', required=True, help='Path to base_model_game_data_with_rolling.csv')
    
    args = parser.parse_args()
    
    print("Loading data...")
    teams, games, rank_lookup = load_data(args.team_data, args.game_data)
    print(f"  Teams with ranks: {len(rank_lookup)}")
    print(f"  Games: {len(games)}")
    
    # Add quad levels to games (expands to 2 rows per game)
    print("\nCalculating game quad levels...")
    games = add_game_quads(games, rank_lookup)
    valid_games = games.dropna(subset=['game_quad'])
    print(f"  Game rows with quad data: {len(valid_games)} (2 per game)")
    
    # Calculate team quad records
    print("\nCalculating team quad records...")
    team_records = calculate_team_quad_records(games)
    print(f"  Teams with records: {len(team_records)}")
    
    # Show some example records
    print("\n" + "-" * 70)
    print("SAMPLE TEAM QUAD RECORDS (Top 10 by rank)")
    print("-" * 70)
    
    top_teams = sorted(rank_lookup.items(), key=lambda x: x[1])[:10]
    print(f"\n{'Team':<8} {'Rank':>6} {'Q1 W-L':>10} {'Q2 W-L':>10} {'Q3 W-L':>10} {'Q4 W-L':>10} {'Quality':>8}")
    print("-" * 70)
    
    for team, rank in top_teams:
        if team in team_records:
            rec = team_records[team]
            q1 = f"{rec['q1_wins']}-{rec['q1_losses']}"
            q2 = f"{rec['q2_wins']}-{rec['q2_losses']}"
            q3 = f"{rec['q3_wins']}-{rec['q3_losses']}"
            q4 = f"{rec['q4_wins']}-{rec['q4_losses']}"
            print(f"{team:<8} {rank:>6} {q1:>10} {q2:>10} {q3:>10} {q4:>10} {rec['quality_score']:>8}")
    
    # Run analyses
    df = analyze_quad_correlations(games, team_records, rank_lookup)
    analyze_by_game_quad(games, team_records, rank_lookup)
    analyze_q1_vs_q4_teams(games, team_records, rank_lookup)
    recommend_features(df)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()