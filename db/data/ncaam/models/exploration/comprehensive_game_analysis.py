"""
Extended Deep Analysis - Part 2

Continuing to dig into the data. Building on findings from Part 1.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = '../../processed/'

def load_data():
    games = pd.read_csv(f'{PROCESSED_DIR}game_analysis_export.csv')
    teams = pd.read_csv(f'{PROCESSED_DIR}ncaam_team_data_final.csv')
    
    team_lookup = teams.set_index('Team_Code').to_dict('index')
    
    def get_stat(code, stat):
        return team_lookup.get(code, {}).get(stat, None)
    
    # Add team stats
    for prefix, col in [('h', 'home_team_code'), ('a', 'away_team_code')]:
        games[f'{prefix}_quality'] = games[col].apply(lambda x: get_stat(x, 'Weighted Quality'))
        games[f'{prefix}_off_rank'] = games[col].apply(lambda x: get_stat(x, 'Adj. Off. Eff Rank'))
        games[f'{prefix}_def_rank'] = games[col].apply(lambda x: get_stat(x, 'Adj. Def. Eff Rank'))
        games[f'{prefix}_adj_off'] = games[col].apply(lambda x: get_stat(x, 'Adj. Off. Eff'))
        games[f'{prefix}_adj_def'] = games[col].apply(lambda x: get_stat(x, 'Adj. Def. Eff'))
        games[f'{prefix}_efg_season'] = games[col].apply(lambda x: get_stat(x, 'Eff. FG% Off'))
        games[f'{prefix}_efg_def_season'] = games[col].apply(lambda x: get_stat(x, 'Eff. FG% Def'))
        games[f'{prefix}_tov_season'] = games[col].apply(lambda x: get_stat(x, 'Turnover% Off'))
        games[f'{prefix}_oreb_season'] = games[col].apply(lambda x: get_stat(x, 'Off. Reb%'))
        games[f'{prefix}_tempo'] = games[col].apply(lambda x: get_stat(x, 'Adj. Tempo'))
        games[f'{prefix}_elite_sos'] = games[col].apply(lambda x: get_stat(x, 'Elite SOS'))
        games[f'{prefix}_experience'] = games[col].apply(lambda x: get_stat(x, 'Experience'))
        games[f'{prefix}_talent'] = games[col].apply(lambda x: get_stat(x, 'Talent'))
        games[f'{prefix}_3p_season'] = games[col].apply(lambda x: get_stat(x, '3P% Off'))
        games[f'{prefix}_3p_def_season'] = games[col].apply(lambda x: get_stat(x, '3P% Def'))
        games[f'{prefix}_ft_rate'] = games[col].apply(lambda x: get_stat(x, 'FT Rate Off'))
    
    games['h_combined_rank'] = games['h_off_rank'].fillna(999) + games['h_def_rank'].fillna(999)
    games['a_combined_rank'] = games['a_off_rank'].fillna(999) + games['a_def_rank'].fillna(999)
    games['quality_diff'] = games['h_quality'] - games['a_quality']
    games['rank_gap'] = abs(games['h_combined_rank'] - games['a_combined_rank'])
    
    print(f"Loaded {len(games)} games")
    return games, teams

# =============================================================================
# PART 1: THE DANGER ZONE (Quality Diff 5-20)
# =============================================================================

def analyze_danger_zone(games):
    """Deep dive into quality diff 5-20 where V4 struggles most."""
    print("\n" + "=" * 80)
    print("DANGER ZONE ANALYSIS (Quality Diff 5-20)")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    danger = v4[(abs(v4['quality_diff']) >= 5) & (abs(v4['quality_diff']) <= 20)].copy()
    
    print(f"\nDanger zone games: {len(danger)} ({len(danger)/len(v4):.1%} of all games)")
    print(f"V4 accuracy in danger zone: {danger['v4_correct'].mean():.1%}")
    print(f"V4 accuracy outside danger zone: {v4[~v4.index.isin(danger.index)]['v4_correct'].mean():.1%}")
    
    # Who's the favorite in these games?
    danger['fav_home'] = danger['quality_diff'] > 0
    danger['fav_won'] = ((danger['fav_home'] & (danger['home_won'] == 1)) | 
                         (~danger['fav_home'] & (danger['home_won'] == 0)))
    
    print(f"\nFavorite wins: {danger['fav_won'].mean():.1%}")
    print(f"Home team is favorite: {danger['fav_home'].mean():.1%}")
    
    # What predicts winners in danger zone?
    print("\n--- What predicts winners in DANGER ZONE? ---\n")
    
    # Create differential features
    danger['efg_diff'] = danger['h_efg_pct'] - danger['a_efg_pct']
    danger['tov_diff'] = danger['h_tov_rate'] - danger['a_tov_rate']
    danger['oreb_diff'] = danger['h_oreb_pct'] - danger['a_oreb_pct']
    danger['paint_diff'] = danger['h_pts_in_paint'] - danger['a_pts_in_paint']
    danger['3p_diff'] = danger['h_3p_pct'] - danger['a_3p_pct']
    danger['off_rating_diff'] = danger['h_off_rating'] - danger['a_off_rating']
    danger['fast_break_diff'] = danger['h_fast_break_pts'] - danger['a_fast_break_pts']
    
    # Season-level diffs
    danger['season_efg_diff'] = danger['h_efg_season'] - danger['a_efg_season']
    danger['season_3p_diff'] = danger['h_3p_season'] - danger['a_3p_season']
    danger['talent_diff'] = danger['h_talent'] - danger['a_talent']
    danger['experience_diff'] = danger['h_experience'] - danger['a_experience']
    danger['tempo_diff'] = danger['h_tempo'] - danger['a_tempo']
    
    features = [
        ('efg_diff', 'Game eFG% Diff'),
        ('tov_diff', 'Game TOV Rate Diff'),
        ('oreb_diff', 'Game OREB% Diff'),
        ('paint_diff', 'Pts in Paint Diff'),
        ('3p_diff', 'Game 3P% Diff'),
        ('off_rating_diff', 'Off Rating Diff'),
        ('fast_break_diff', 'Fast Break Diff'),
        ('season_efg_diff', 'Season eFG% Diff'),
        ('season_3p_diff', 'Season 3P% Diff'),
        ('talent_diff', 'Talent Diff'),
        ('experience_diff', 'Experience Diff'),
        ('tempo_diff', 'Tempo Diff'),
        ('quality_diff', 'Quality Diff'),
        ('rank_gap', 'Rank Gap'),
    ]
    
    print(f"{'Feature':<25} {'Corr w/Home Win':>15} {'Win Mean':>12} {'Lose Mean':>12}")
    print("-" * 70)
    
    results = []
    for col, name in features:
        if col in danger.columns:
            corr = danger[col].corr(danger['home_won'].astype(float))
            win_mean = danger[danger['home_won'] == 1][col].mean()
            lose_mean = danger[danger['home_won'] == 0][col].mean()
            results.append((name, corr, win_mean, lose_mean))
    
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr, win, lose in results:
        print(f"{name:<25} {corr:>+15.3f} {win:>+12.1f} {lose:>+12.1f}")
    
    # V4 wrong in danger zone - what's happening?
    print("\n--- V4 WRONG in Danger Zone ---\n")
    wrong = danger[danger['v4_correct'] == 0]
    right = danger[danger['v4_correct'] == 1]
    
    print(f"Wrong: {len(wrong)}, Right: {len(right)}")
    
    print(f"\n{'Metric':<25} {'Right':>12} {'Wrong':>12} {'Diff':>12}")
    print("-" * 60)
    
    for col in ['v4_confidence', 'efg_diff', '3p_diff', 'paint_diff', 'quality_diff', 'rank_gap']:
        if col in danger.columns:
            r = right[col].mean()
            w = wrong[col].mean()
            print(f"{col:<25} {r:>+12.2f} {w:>+12.2f} {w-r:>+12.2f}")
    
    return danger

# =============================================================================
# PART 2: HOME COURT ADVANTAGE DEEP DIVE
# =============================================================================

def analyze_home_court(games):
    """When does home court actually matter?"""
    print("\n" + "=" * 80)
    print("HOME COURT ADVANTAGE ANALYSIS")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    
    print(f"\nOverall home win rate: {v4['home_won'].mean():.1%}")
    print(f"Average home margin: {v4['home_margin'].mean():+.1f}")
    
    # Home court by quality comparison
    print("\n--- Home Court Value by Quality Matchup ---\n")
    
    v4['home_is_better'] = v4['quality_diff'] > 0
    v4['away_is_better'] = v4['quality_diff'] < 0
    v4['teams_equal'] = abs(v4['quality_diff']) <= 5
    
    print(f"{'Scenario':<35} {'Games':>8} {'Home Win%':>12} {'Avg Margin':>12}")
    print("-" * 70)
    
    scenarios = [
        ('Home much better (qual +30)', v4['quality_diff'] >= 30),
        ('Home better (qual +10 to +30)', (v4['quality_diff'] >= 10) & (v4['quality_diff'] < 30)),
        ('Home slightly better (qual +5 to +10)', (v4['quality_diff'] >= 5) & (v4['quality_diff'] < 10)),
        ('Teams equal (qual ±5)', abs(v4['quality_diff']) <= 5),
        ('Away slightly better (qual -5 to -10)', (v4['quality_diff'] <= -5) & (v4['quality_diff'] > -10)),
        ('Away better (qual -10 to -30)', (v4['quality_diff'] <= -10) & (v4['quality_diff'] > -30)),
        ('Away much better (qual -30)', v4['quality_diff'] <= -30),
    ]
    
    for name, mask in scenarios:
        subset = v4[mask]
        if len(subset) > 20:
            print(f"{name:<35} {len(subset):>8} {subset['home_won'].mean():>11.1%} {subset['home_margin'].mean():>+11.1f}")
    
    # Does home court help bad teams more?
    print("\n--- Home Court Boost by Home Team Quality ---\n")
    
    def get_tier(rank):
        if rank <= 75: return 'Good (≤75)'
        elif rank <= 150: return 'Avg (76-150)'
        elif rank <= 300: return 'Below (151-300)'
        else: return 'Poor (300+)'
    
    v4['h_tier'] = v4['h_combined_rank'].apply(get_tier)
    
    print(f"{'Home Tier':<20} {'Games':>8} {'Home Win%':>12} {'When Underdog':>15}")
    print("-" * 60)
    
    for tier in ['Good (≤75)', 'Avg (76-150)', 'Below (151-300)', 'Poor (300+)']:
        subset = v4[v4['h_tier'] == tier]
        underdog = subset[subset['quality_diff'] < 0]
        if len(subset) > 30:
            und_win = underdog['home_won'].mean() if len(underdog) > 10 else float('nan')
            print(f"{tier:<20} {len(subset):>8} {subset['home_won'].mean():>11.1%} {und_win:>14.1%}")

# =============================================================================
# PART 3: 3-POINT SHOOTING DEEP DIVE
# =============================================================================

def analyze_three_point(games):
    """3P% showed up as important in close games. Dig deeper."""
    print("\n" + "=" * 80)
    print("3-POINT SHOOTING ANALYSIS")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    
    # Game 3P% vs season 3P%
    v4['h_3p_vs_season'] = v4['h_3p_pct'] - v4['h_3p_season']
    v4['a_3p_vs_season'] = v4['a_3p_pct'] - v4['a_3p_season']
    
    print("\n--- 3P% Variance from Season Average ---")
    print(f"Home: mean = {v4['h_3p_vs_season'].mean():+.1f}, std = {v4['h_3p_vs_season'].std():.1f}")
    print(f"Away: mean = {v4['a_3p_vs_season'].mean():+.1f}, std = {v4['a_3p_vs_season'].std():.1f}")
    
    # When home team gets hot from 3
    print("\n--- Impact of Home Team 3P% vs Season ---\n")
    print(f"{'3P Performance':<25} {'Games':>8} {'Home Win%':>12} {'V4 Acc':>10}")
    print("-" * 60)
    
    for name, cond in [
        ('Cold (<-10%)', v4['h_3p_vs_season'] < -10),
        ('Below avg (-10 to -5%)', (v4['h_3p_vs_season'] >= -10) & (v4['h_3p_vs_season'] < -5)),
        ('Slightly below (-5 to 0%)', (v4['h_3p_vs_season'] >= -5) & (v4['h_3p_vs_season'] < 0)),
        ('Slightly above (0 to 5%)', (v4['h_3p_vs_season'] >= 0) & (v4['h_3p_vs_season'] < 5)),
        ('Above avg (5 to 10%)', (v4['h_3p_vs_season'] >= 5) & (v4['h_3p_vs_season'] < 10)),
        ('Hot (>10%)', v4['h_3p_vs_season'] >= 10),
    ]:
        subset = v4[cond]
        if len(subset) > 30:
            print(f"{name:<25} {len(subset):>8} {subset['home_won'].mean():>11.1%} {subset['v4_correct'].mean():>9.1%}")
    
    # 3P differential impact
    print("\n--- 3P% Differential (Home - Away) Impact ---\n")
    v4['3p_diff'] = v4['h_3p_pct'] - v4['a_3p_pct']
    
    print(f"{'3P Diff':<25} {'Games':>8} {'Home Win%':>12} {'V4 Acc':>10}")
    print("-" * 60)
    
    for name, cond in [
        ('Home much worse (<-15)', v4['3p_diff'] < -15),
        ('Home worse (-15 to -5)', (v4['3p_diff'] >= -15) & (v4['3p_diff'] < -5)),
        ('Close (-5 to +5)', (v4['3p_diff'] >= -5) & (v4['3p_diff'] <= 5)),
        ('Home better (+5 to +15)', (v4['3p_diff'] > 5) & (v4['3p_diff'] <= 15)),
        ('Home much better (>+15)', v4['3p_diff'] > 15),
    ]:
        subset = v4[cond]
        if len(subset) > 30:
            print(f"{name:<25} {len(subset):>8} {subset['home_won'].mean():>11.1%} {subset['v4_correct'].mean():>9.1%}")
    
    # In close games specifically
    print("\n--- 3P% Impact in CLOSE GAMES (margin ≤5) ---\n")
    close = v4[abs(v4['home_margin']) <= 5]
    
    print(f"{'3P Diff':<25} {'Games':>8} {'Home Win%':>12}")
    print("-" * 50)
    
    for name, cond in [
        ('Home worse (<-5)', close['3p_diff'] < -5),
        ('Close (-5 to +5)', (close['3p_diff'] >= -5) & (close['3p_diff'] <= 5)),
        ('Home better (>+5)', close['3p_diff'] > 5),
    ]:
        subset = close[cond]
        if len(subset) > 20:
            print(f"{name:<25} {len(subset):>8} {subset['home_won'].mean():>11.1%}")

# =============================================================================
# PART 4: MATCHUP SPECIFIC ANALYSIS - OFFENSE VS DEFENSE
# =============================================================================

def analyze_matchups(games):
    """How does offensive strength vs defensive strength play out?"""
    print("\n" + "=" * 80)
    print("OFFENSE VS DEFENSE MATCHUP ANALYSIS")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    
    # Calculate matchup features
    # Home offense vs away defense
    v4['h_off_vs_a_def'] = v4['h_adj_off'] - v4['a_adj_def']  # Positive = home offense better than away D
    v4['a_off_vs_h_def'] = v4['a_adj_off'] - v4['h_adj_def']  # Positive = away offense better than home D
    
    # Who has the bigger matchup advantage?
    v4['matchup_edge'] = v4['h_off_vs_a_def'] - v4['a_off_vs_h_def']
    
    print("\n--- Matchup Edge (Home off vs Away def) - (Away off vs Home def) ---\n")
    print(f"{'Matchup Edge':<30} {'Games':>8} {'Home Win%':>12} {'V4 Acc':>10}")
    print("-" * 65)
    
    for name, cond in [
        ('Big away edge (<-15)', v4['matchup_edge'] < -15),
        ('Away edge (-15 to -5)', (v4['matchup_edge'] >= -15) & (v4['matchup_edge'] < -5)),
        ('Slight away (-5 to 0)', (v4['matchup_edge'] >= -5) & (v4['matchup_edge'] < 0)),
        ('Slight home (0 to +5)', (v4['matchup_edge'] >= 0) & (v4['matchup_edge'] < 5)),
        ('Home edge (+5 to +15)', (v4['matchup_edge'] >= 5) & (v4['matchup_edge'] < 15)),
        ('Big home edge (>+15)', v4['matchup_edge'] >= 15),
    ]:
        subset = v4[cond]
        if len(subset) > 30:
            print(f"{name:<30} {len(subset):>8} {subset['home_won'].mean():>11.1%} {subset['v4_correct'].mean():>9.1%}")
    
    # Does this differ from quality_diff?
    print("\n--- Matchup Edge vs Quality Diff Correlation ---")
    corr = v4['matchup_edge'].corr(v4['quality_diff'])
    print(f"Correlation: {corr:.3f}")
    
    # When matchup edge disagrees with quality diff
    print("\n--- When Matchup Edge Disagrees with Quality Diff ---\n")
    
    # Quality says home, matchup says away
    disagree1 = v4[(v4['quality_diff'] > 5) & (v4['matchup_edge'] < -5)]
    print(f"Quality favors home (+5), Matchup favors away (-5): {len(disagree1)} games")
    if len(disagree1) > 10:
        print(f"  Home wins: {disagree1['home_won'].mean():.1%}")
        print(f"  V4 accuracy: {disagree1['v4_correct'].mean():.1%}")
    
    # Quality says away, matchup says home
    disagree2 = v4[(v4['quality_diff'] < -5) & (v4['matchup_edge'] > 5)]
    print(f"\nQuality favors away (-5), Matchup favors home (+5): {len(disagree2)} games")
    if len(disagree2) > 10:
        print(f"  Home wins: {disagree2['home_won'].mean():.1%}")
        print(f"  V4 accuracy: {disagree2['v4_correct'].mean():.1%}")

# =============================================================================
# PART 5: TEMPO IMPACT
# =============================================================================

def analyze_tempo(games):
    """Does tempo mismatch matter?"""
    print("\n" + "=" * 80)
    print("TEMPO ANALYSIS")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    
    v4['tempo_diff'] = v4['h_tempo'] - v4['a_tempo']
    v4['avg_tempo'] = (v4['h_tempo'] + v4['a_tempo']) / 2
    
    print(f"\nAverage game pace: {v4['pace'].mean():.1f}")
    print(f"Pace std: {v4['pace'].std():.1f}")
    
    # Does the faster team win?
    print("\n--- Does Tempo Preference Predict Wins? ---\n")
    
    # When home wants faster pace
    print(f"{'Tempo Diff (H-A)':<25} {'Games':>8} {'Home Win%':>12} {'V4 Acc':>10}")
    print("-" * 60)
    
    for name, cond in [
        ('Home much slower (<-5)', v4['tempo_diff'] < -5),
        ('Home slower (-5 to 0)', (v4['tempo_diff'] >= -5) & (v4['tempo_diff'] < 0)),
        ('Home faster (0 to +5)', (v4['tempo_diff'] >= 0) & (v4['tempo_diff'] < 5)),
        ('Home much faster (>+5)', v4['tempo_diff'] >= 5),
    ]:
        subset = v4[cond]
        if len(subset) > 30:
            print(f"{name:<25} {len(subset):>8} {subset['home_won'].mean():>11.1%} {subset['v4_correct'].mean():>9.1%}")
    
    # Who controls the pace?
    print("\n--- Does Home Team Control Pace? ---")
    v4['pace_vs_h_pref'] = v4['pace'] - v4['h_tempo']
    v4['pace_vs_a_pref'] = v4['pace'] - v4['a_tempo']
    
    print(f"Actual pace vs Home preferred: {v4['pace_vs_h_pref'].mean():+.1f}")
    print(f"Actual pace vs Away preferred: {v4['pace_vs_a_pref'].mean():+.1f}")
    
    # In close games
    print("\n--- Tempo Impact in Close Games ---")
    close = v4[abs(v4['home_margin']) <= 5]
    corr = close['tempo_diff'].corr(close['home_won'].astype(float))
    print(f"Tempo diff correlation with home win: {corr:.3f}")

# =============================================================================
# PART 6: EXPERIENCE AND TALENT
# =============================================================================

def analyze_experience_talent(games):
    """Do experience and talent matter beyond quality?"""
    print("\n" + "=" * 80)
    print("EXPERIENCE AND TALENT ANALYSIS")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    
    v4['talent_diff'] = v4['h_talent'] - v4['a_talent']
    v4['exp_diff'] = v4['h_experience'] - v4['a_experience']
    
    print("\n--- Talent Diff Impact ---\n")
    print(f"{'Talent Diff':<25} {'Games':>8} {'Home Win%':>12} {'V4 Acc':>10}")
    print("-" * 60)
    
    for name, cond in [
        ('Home much less (<-50)', v4['talent_diff'] < -50),
        ('Home less (-50 to -10)', (v4['talent_diff'] >= -50) & (v4['talent_diff'] < -10)),
        ('Similar (-10 to +10)', (v4['talent_diff'] >= -10) & (v4['talent_diff'] <= 10)),
        ('Home more (+10 to +50)', (v4['talent_diff'] > 10) & (v4['talent_diff'] <= 50)),
        ('Home much more (>+50)', v4['talent_diff'] > 50),
    ]:
        subset = v4[cond]
        if len(subset) > 30:
            print(f"{name:<25} {len(subset):>8} {subset['home_won'].mean():>11.1%} {subset['v4_correct'].mean():>9.1%}")
    
    # Correlation with quality_diff
    print(f"\nTalent diff correlation with quality_diff: {v4['talent_diff'].corr(v4['quality_diff']):.3f}")
    
    # Experience in close games
    print("\n--- Experience Impact in Close Games ---")
    close = v4[abs(v4['home_margin']) <= 5]
    exp_corr = close['exp_diff'].corr(close['home_won'].astype(float))
    talent_corr = close['talent_diff'].corr(close['home_won'].astype(float))
    print(f"Experience diff correlation with home win: {exp_corr:.3f}")
    print(f"Talent diff correlation with home win: {talent_corr:.3f}")
    
    # When experience disagrees with quality
    print("\n--- When Experience Disagrees with Quality ---")
    # Home has less quality but more experience
    disagree = v4[(v4['quality_diff'] < -5) & (v4['exp_diff'] > 0.3)]
    print(f"\nHome is underdog (quality -5) but more experienced: {len(disagree)} games")
    if len(disagree) > 20:
        print(f"  Home wins: {disagree['home_won'].mean():.1%}")
        expected = v4[v4['quality_diff'] < -5]['home_won'].mean()
        print(f"  Expected (all quality -5 games): {expected:.1%}")

# =============================================================================
# PART 7: LOW CONFIDENCE DEEP DIVE
# =============================================================================

def analyze_low_confidence(games):
    """Deep dive into the coin flip zone."""
    print("\n" + "=" * 80)
    print("LOW CONFIDENCE (<60%) DEEP DIVE")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    low_conf = v4[v4['v4_confidence'] < 60].copy()
    
    print(f"\nLow confidence games: {len(low_conf)}")
    print(f"V4 accuracy: {low_conf['v4_correct'].mean():.1%}")
    print(f"Home win rate: {low_conf['home_won'].mean():.1%}")
    
    # What predicts winners in low confidence games?
    print("\n--- What Predicts Winners in Low Confidence Games? ---\n")
    
    low_conf['efg_diff'] = low_conf['h_efg_pct'] - low_conf['a_efg_pct']
    low_conf['3p_diff'] = low_conf['h_3p_pct'] - low_conf['a_3p_pct']
    low_conf['paint_diff'] = low_conf['h_pts_in_paint'] - low_conf['a_pts_in_paint']
    low_conf['off_rating_diff'] = low_conf['h_off_rating'] - low_conf['a_off_rating']
    
    features = ['quality_diff', 'rank_gap', 'efg_diff', '3p_diff', 'paint_diff', 
                'off_rating_diff', 'h_3p_pct', 'a_3p_pct', 'pace']
    
    print(f"{'Feature':<25} {'Corr w/Home Win':>15}")
    print("-" * 45)
    
    for feat in features:
        if feat in low_conf.columns:
            corr = low_conf[feat].corr(low_conf['home_won'].astype(float))
            print(f"{feat:<25} {corr:>+15.3f}")
    
    # V4 right vs wrong in low conf
    print("\n--- Right vs Wrong in Low Confidence ---\n")
    right = low_conf[low_conf['v4_correct'] == 1]
    wrong = low_conf[low_conf['v4_correct'] == 0]
    
    print(f"{'Metric':<20} {'Right':>12} {'Wrong':>12}")
    print("-" * 50)
    
    for col in ['quality_diff', 'rank_gap', 'home_margin']:
        r = abs(right[col]).mean()
        w = abs(wrong[col]).mean()
        print(f"{col:<20} {r:>12.1f} {w:>12.1f}")
    
    # Should we just pick home in low conf?
    print("\n--- Simple Rules in Low Confidence ---")
    print(f"Always pick home: {low_conf['home_won'].mean():.1%}")
    print(f"Always pick favorite (by quality): {(((low_conf['quality_diff'] > 0) & (low_conf['home_won'] == 1)) | ((low_conf['quality_diff'] < 0) & (low_conf['home_won'] == 0))).mean():.1%}")
    print(f"V4's accuracy: {low_conf['v4_correct'].mean():.1%}")

# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def generate_recommendations(games):
    """Generate specific recommendations based on all analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("=" * 80)
    
    print("""
Based on the deep analysis, here are specific recommendations:

================================================================================
1. FEATURE ENGINEERING
================================================================================

A. ADD THESE NEW FEATURES:
   - matchup_edge = (h_adj_off - a_adj_def) - (a_adj_off - h_adj_def)
     * Captures offense vs defense matchup beyond raw quality
   
   - h_3p_consistency / a_3p_consistency (std of 3P% in recent games)
     * High variance teams are upset risks
   
   - quality_gap_bucket (categorical: 0-5, 5-15, 15-30, 30+)
     * Different dynamics in each bucket
   
   - is_danger_zone = 1 if quality_diff between 5-20
     * V4 struggles here, needs special handling

B. CONDITIONAL FEATURES FOR CLOSE GAMES:
   - 3p_diff_close = 3p_diff * (1 if predicted_margin < 5 else 0)
   - efg_diff_close = efg_diff * (1 if predicted_margin < 5 else 0)
   
C. REDUCE WEIGHT OF THESE IN CLOSE GAMES:
   - TOV rate (correlation drops from -0.214 to -0.005 in close games)
   - OREB% (correlation drops from +0.307 to +0.136 in close games)

================================================================================
2. MODEL ARCHITECTURE
================================================================================

A. CONSIDER SEPARATE MODELS OR BRANCHES:
   - One for mismatches (quality_diff > 30): Current model is 86%+ accurate
   - One for close matchups (quality_diff < 15): Need different feature weights
   
B. CONFIDENCE THRESHOLD:
   - Below 55% confidence, V4 is 50.5% accurate (coin flip)
   - Consider: Don't make predictions below 55%, or flag as "no pick"

C. CALIBRATION ADJUSTMENT:
   - V4 is overconfident in the 50-65% range
   - Actual accuracy is ~10% lower than confidence in this range

================================================================================
3. SPECIFIC FIXES
================================================================================

A. DANGER ZONE (quality_diff 5-20):
   - 588 games, only 69.6% accurate
   - In this zone: 3P% diff matters MORE, quality_diff matters LESS
   - Consider boosting 3P% weight when quality_diff is in this range

B. HOME UNDERDOG WITH EXPERIENCE:
   - When home team is underdog but more experienced, they overperform
   - Add interaction: exp_diff * (1 if quality_diff < 0 else 0)

C. TEMPO CONTROL:
   - Home team generally controls pace (actual closer to home preferred)
   - When home prefers fast and plays fast, they win more

================================================================================
4. DATA COLLECTION
================================================================================

A. NEED TO TRACK:
   - Team consistency/variance metrics (shooting variance, margin variance)
   - Recent form (last 5-10 games performance)
   - Head-to-head history
   - Injury/lineup data if available

B. GAME-LEVEL FEATURES TO ADD:
   - Conference game indicator
   - Days rest for each team
   - Travel distance

================================================================================
5. BETTING STRATEGY IMPLICATIONS
================================================================================

A. HIGH CONFIDENCE (70%+):
   - V4 is 86%+ accurate here
   - These are solid picks

B. LOW CONFIDENCE (<60%):
   - V4 is barely better than coin flip
   - Consider passing on these or using alternative signals

C. DANGER ZONE (quality_diff 5-20):
   - High upset rate (37-43%)
   - Look for 3P shooting matchup edge
   - Look for experience edge

""")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EXTENDED DEEP ANALYSIS - PART 2")
    print("=" * 80)
    
    games, teams = load_data()
    
    analyze_danger_zone(games)
    analyze_home_court(games)
    analyze_three_point(games)
    analyze_matchups(games)
    analyze_tempo(games)
    analyze_experience_talent(games)
    analyze_low_confidence(games)
    generate_recommendations(games)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()