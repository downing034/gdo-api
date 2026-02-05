"""
V4 Error Analysis - Find the Biggest Opportunity

V4 is at 77.5% (1195/1541). We need 80% (1233/1541).
That's 38 more correct predictions needed.

This script analyzes the 346 errors to find patterns and test which
proposed features would have helped most.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = '../../processed/'

def load_data():
    """Load all data sources."""
    # Game-level box scores
    games = pd.read_csv(f'{PROCESSED_DIR}game_analysis_export.csv')
    
    # Team season stats
    teams = pd.read_csv(f'{PROCESSED_DIR}ncaam_team_data_final.csv')
    
    print(f"Games: {len(games)}")
    print(f"Teams: {len(teams)}")
    
    return games, teams

def add_team_stats(games, teams):
    """Add team season stats to games."""
    team_lookup = teams.set_index('Team_Code').to_dict('index')
    
    def get_stat(code, stat):
        return team_lookup.get(code, {}).get(stat, None)
    
    stats_to_add = [
        ('quality', 'Weighted Quality'),
        ('adj_off', 'Adj. Off. Eff'),
        ('adj_def', 'Adj. Def. Eff'),
        ('off_rank', 'Adj. Off. Eff Rank'),
        ('def_rank', 'Adj. Def. Eff Rank'),
        ('tempo', 'Adj. Tempo'),
        ('3p_off', '3P% Off'),
        ('3p_def', '3P% Def'),
        ('3p_rate', '3P Rate Off'),
        ('tov_off', 'Turnover% Off'),
        ('tov_def', 'Turnover% Def'),
        ('oreb', 'Off. Reb%'),
        ('dreb', 'Def. Reb%'),
        ('efg_off', 'Eff. FG% Off'),
        ('efg_def', 'Eff. FG% Def'),
        ('ftr_off', 'FT Rate Off'),
        ('ftr_def', 'FT Rate Def'),
        ('experience', 'Experience'),
        ('talent', 'Talent'),
    ]
    
    for prefix, col in [('h', 'home_team_code'), ('a', 'away_team_code')]:
        for stat_name, stat_col in stats_to_add:
            games[f'{prefix}_{stat_name}'] = games[col].apply(lambda x: get_stat(x, stat_col))
    
    # Derived features
    games['quality_diff'] = games['h_quality'] - games['a_quality']
    games['h_combined_rank'] = games['h_off_rank'].fillna(999) + games['h_def_rank'].fillna(999)
    games['a_combined_rank'] = games['a_off_rank'].fillna(999) + games['a_def_rank'].fillna(999)
    games['rank_gap'] = abs(games['h_combined_rank'] - games['a_combined_rank'])
    
    return games

def build_proposed_features(games):
    """Build all proposed features to test which would help."""
    
    # === VARIANCE FEATURES ===
    # 3P volatility: reliance on 3s × miss rate
    games['h_vol3'] = games['h_3p_rate'] * (1 - games['h_3p_off']/100)
    games['a_vol3'] = games['a_3p_rate'] * (1 - games['a_3p_off']/100)
    games['vol3_sum'] = games['h_vol3'] + games['a_vol3']
    games['vol3_diff'] = games['h_vol3'] - games['a_vol3']
    
    # Turnover chaos
    games['to_pressure_home'] = games['a_tov_off'] * games['h_tov_def']
    games['to_pressure_away'] = games['h_tov_off'] * games['a_tov_def']
    games['to_pressure_diff'] = games['to_pressure_home'] - games['to_pressure_away']
    games['chaos_index'] = games['h_tov_off'] + games['a_tov_off'] + games['h_tov_def'] + games['a_tov_def']
    
    # Low possession flag
    games['expected_tempo'] = (games['h_tempo'] + games['a_tempo']) / 2
    games['low_poss_flag'] = (games['expected_tempo'] < 65).astype(float)
    games['quality_x_low_poss'] = games['quality_diff'] * games['low_poss_flag']
    
    # === GAME CONTROL FEATURES (from box scores) ===
    games['time_control_diff'] = games['h_time_leading_pct'] - games['a_time_leading_pct']
    games['lead_strength_diff'] = games['h_largest_lead'] - games['a_largest_lead']
    
    # === STYLE FEATURES ===
    # Floor (consistency) vs Ceiling (upside)
    games['h_floor'] = (1 - games['h_tov_off']/100) * (1 - games['h_ftr_def']/100)
    games['a_floor'] = (1 - games['a_tov_off']/100) * (1 - games['a_ftr_def']/100)
    games['floor_diff'] = games['h_floor'] - games['a_floor']
    
    games['h_ceiling'] = games['h_3p_rate'] * games['h_3p_off'] + games['h_ftr_off']
    games['a_ceiling'] = games['a_3p_rate'] * games['a_3p_off'] + games['a_ftr_off']
    games['ceiling_diff'] = games['h_ceiling'] - games['a_ceiling']
    
    # === 3P MATCHUP (offense vs defense) ===
    games['h_3p_vs_def'] = games['h_3p_off'] - games['a_3p_def']
    games['a_3p_vs_def'] = games['a_3p_off'] - games['h_3p_def']
    games['threep_matchup_adv'] = games['h_3p_vs_def'] - games['a_3p_vs_def']
    
    # === REBOUNDING MATCHUP ===
    games['reb_matchup'] = (games['h_oreb'] + (100 - games['a_dreb'])) - (games['a_oreb'] + (100 - games['h_dreb']))
    
    # === DANGER ZONE ===
    abs_qd = abs(games['quality_diff'])
    games['is_danger_zone'] = ((abs_qd >= 5) & (abs_qd <= 20)).astype(float)
    games['is_mismatch'] = (abs_qd > 30).astype(float)
    games['is_tossup'] = (abs_qd < 5).astype(float)
    
    return games

def analyze_v4_errors(games):
    """Deep analysis of V4 errors."""
    print("\n" + "=" * 80)
    print("V4 ERROR ANALYSIS")
    print("=" * 80)
    
    v4 = games[games['v4_correct'].notna()].copy()
    wrong = v4[v4['v4_correct'] == 0]
    right = v4[v4['v4_correct'] == 1]
    
    print(f"\nTotal V4 predictions: {len(v4)}")
    print(f"Correct: {len(right)} ({len(right)/len(v4):.1%})")
    print(f"Wrong: {len(wrong)} ({len(wrong)/len(v4):.1%})")
    
    # === WHERE ARE THE ERRORS? ===
    print("\n" + "-" * 60)
    print("WHERE ARE THE ERRORS?")
    print("-" * 60)
    
    # By confidence
    print("\n--- By V4 Confidence ---")
    print(f"{'Confidence':<15} {'Total':>8} {'Wrong':>8} {'Error%':>10} {'% of Errors':>12}")
    print("-" * 55)
    
    error_by_conf = []
    for low, high in [(50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 100)]:
        subset = v4[(v4['v4_confidence'] >= low) & (v4['v4_confidence'] < high)]
        wrong_n = (subset['v4_correct'] == 0).sum()
        if len(subset) > 0:
            pct_of_errors = wrong_n / len(wrong) * 100
            error_by_conf.append((f"{low}-{high}%", len(subset), wrong_n, wrong_n/len(subset)*100, pct_of_errors))
            print(f"{low}-{high}%{'':<8} {len(subset):>8} {wrong_n:>8} {wrong_n/len(subset):>9.1%} {pct_of_errors:>11.1f}%")
    
    # By quality diff zone
    print("\n--- By Quality Diff Zone ---")
    print(f"{'Zone':<20} {'Total':>8} {'Wrong':>8} {'Error%':>10} {'% of Errors':>12}")
    print("-" * 60)
    
    zones = [
        ('Tossup (<5)', v4['is_tossup'] == 1),
        ('Danger (5-20)', v4['is_danger_zone'] == 1),
        ('Medium (20-30)', (abs(v4['quality_diff']) >= 20) & (abs(v4['quality_diff']) < 30)),
        ('Mismatch (>30)', v4['is_mismatch'] == 1),
    ]
    
    for name, mask in zones:
        subset = v4[mask]
        wrong_n = (subset['v4_correct'] == 0).sum()
        if len(subset) > 0:
            pct_of_errors = wrong_n / len(wrong) * 100
            print(f"{name:<20} {len(subset):>8} {wrong_n:>8} {wrong_n/len(subset):>9.1%} {pct_of_errors:>11.1f}%")
    
    # By actual margin
    print("\n--- By Actual Game Margin ---")
    print(f"{'Margin':<20} {'Total':>8} {'Wrong':>8} {'Error%':>10} {'% of Errors':>12}")
    print("-" * 60)
    
    margins = [
        ('Blowout (>15)', abs(v4['home_margin']) > 15),
        ('Clear (10-15)', (abs(v4['home_margin']) > 10) & (abs(v4['home_margin']) <= 15)),
        ('Medium (5-10)', (abs(v4['home_margin']) > 5) & (abs(v4['home_margin']) <= 10)),
        ('Close (<=5)', abs(v4['home_margin']) <= 5),
    ]
    
    for name, mask in margins:
        subset = v4[mask]
        wrong_n = (subset['v4_correct'] == 0).sum()
        if len(subset) > 0:
            pct_of_errors = wrong_n / len(wrong) * 100
            print(f"{name:<20} {len(subset):>8} {wrong_n:>8} {wrong_n/len(subset):>9.1%} {pct_of_errors:>11.1f}%")
    
    return v4, wrong, right

def test_proposed_features(v4, wrong, right):
    """Test which proposed features would have helped most."""
    print("\n" + "=" * 80)
    print("TESTING PROPOSED FEATURES")
    print("=" * 80)
    
    print("\n--- Which features differ most between Right and Wrong? ---\n")
    
    features_to_test = [
        # Variance features
        ('vol3_sum', 'Total 3P volatility'),
        ('vol3_diff', '3P volatility diff'),
        ('to_pressure_diff', 'TO pressure diff'),
        ('chaos_index', 'Chaos index'),
        ('expected_tempo', 'Expected tempo'),
        ('low_poss_flag', 'Low possession flag'),
        
        # Game control (note: these are outcome data, for analysis only)
        ('time_control_diff', 'Time control diff'),
        ('lead_strength_diff', 'Lead strength diff'),
        
        # Style features
        ('floor_diff', 'Floor (consistency) diff'),
        ('ceiling_diff', 'Ceiling (upside) diff'),
        
        # Matchup features
        ('threep_matchup_adv', '3P matchup advantage'),
        ('reb_matchup', 'Rebounding matchup'),
        
        # Existing features for comparison
        ('quality_diff', 'Quality diff'),
        ('rank_gap', 'Rank gap'),
        ('v4_confidence', 'V4 confidence'),
    ]
    
    print(f"{'Feature':<25} {'Right Mean':>12} {'Wrong Mean':>12} {'Diff':>10} {'p-value':>10}")
    print("-" * 75)
    
    results = []
    for col, name in features_to_test:
        if col in v4.columns:
            r_data = right[col].dropna()
            w_data = wrong[col].dropna()
            
            if len(r_data) > 10 and len(w_data) > 10:
                r_mean = r_data.mean()
                w_mean = w_data.mean()
                
                # For some features, use absolute value
                if col in ['quality_diff', 'vol3_diff', 'to_pressure_diff']:
                    r_mean = abs(r_data).mean()
                    w_mean = abs(w_data).mean()
                
                t_stat, p_val = stats.ttest_ind(r_data, w_data)
                
                results.append((name, r_mean, w_mean, w_mean - r_mean, p_val))
                
                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
                print(f"{name:<25} {r_mean:>12.2f} {w_mean:>12.2f} {w_mean - r_mean:>+10.2f} {p_val:>9.3f} {sig}")
    
    return results

def analyze_biggest_opportunities(v4, wrong):
    """Find the biggest opportunities for improvement."""
    print("\n" + "=" * 80)
    print("BIGGEST OPPORTUNITIES")
    print("=" * 80)
    
    # Overlap analysis: where do errors concentrate?
    print("\n--- Error Concentration (Overlapping Conditions) ---\n")
    
    # Low confidence + close game
    low_conf_close = wrong[(wrong['v4_confidence'] < 60) & (abs(wrong['home_margin']) <= 5)]
    print(f"Low confidence (<60%) AND Close game (<=5): {len(low_conf_close)} errors ({len(low_conf_close)/len(wrong):.1%} of all errors)")
    
    # Danger zone + low confidence
    danger_low = wrong[(wrong['is_danger_zone'] == 1) & (wrong['v4_confidence'] < 65)]
    print(f"Danger zone AND Low confidence (<65%): {len(danger_low)} errors ({len(danger_low)/len(wrong):.1%} of all errors)")
    
    # High confidence wrong (the painful ones)
    high_conf_wrong = wrong[wrong['v4_confidence'] >= 70]
    print(f"High confidence (>=70%) wrong: {len(high_conf_wrong)} errors ({len(high_conf_wrong)/len(wrong):.1%} of all errors)")
    
    # What characterizes high confidence wrong?
    print("\n--- High Confidence Wrong Games (>=70%) ---")
    if len(high_conf_wrong) > 10:
        print(f"Count: {len(high_conf_wrong)}")
        print(f"Avg confidence: {high_conf_wrong['v4_confidence'].mean():.1f}%")
        print(f"Avg quality_diff: {abs(high_conf_wrong['quality_diff']).mean():.1f}")
        print(f"Avg actual margin: {abs(high_conf_wrong['home_margin']).mean():.1f}")
        print(f"In danger zone: {(high_conf_wrong['is_danger_zone'] == 1).sum()}")
        print(f"Mismatches: {(high_conf_wrong['is_mismatch'] == 1).sum()}")
    
    # What would help most?
    print("\n--- Theoretical Improvement Paths ---\n")
    
    # Path 1: Don't predict low confidence
    low_conf_all = v4[v4['v4_confidence'] < 55]
    low_conf_right = (low_conf_all['v4_correct'] == 1).sum()
    low_conf_wrong = (low_conf_all['v4_correct'] == 0).sum()
    print(f"PATH 1: Skip predictions below 55% confidence")
    print(f"  Games skipped: {len(low_conf_all)}")
    print(f"  Errors avoided: {low_conf_wrong}")
    print(f"  Correct predictions lost: {low_conf_right}")
    print(f"  Net change: {low_conf_wrong - low_conf_right:+d}")
    
    # Path 2: Fix danger zone
    danger_all = v4[v4['is_danger_zone'] == 1]
    danger_wrong = (danger_all['v4_correct'] == 0).sum()
    danger_right = (danger_all['v4_correct'] == 1).sum()
    current_acc = danger_right / len(danger_all)
    target_acc = 0.77  # Match overall
    potential_gain = int(len(danger_all) * (target_acc - current_acc))
    print(f"\nPATH 2: Fix danger zone to match overall accuracy")
    print(f"  Danger zone games: {len(danger_all)}")
    print(f"  Current accuracy: {current_acc:.1%}")
    print(f"  If matched 77%: +{potential_gain} correct")
    
    # Path 3: Fix close games
    close_all = v4[abs(v4['home_margin']) <= 5]
    close_wrong = (close_all['v4_correct'] == 0).sum()
    close_right = (close_all['v4_correct'] == 1).sum()
    current_acc = close_right / len(close_all)
    print(f"\nPATH 3: Fix close games")
    print(f"  Close games (margin <=5): {len(close_all)}")
    print(f"  Current accuracy: {current_acc:.1%}")
    print(f"  Errors here: {close_wrong} ({close_wrong/len(wrong):.1%} of all errors)")

def analyze_what_predicts_upsets(v4, wrong):
    """What actually predicts when V4 will be wrong?"""
    print("\n" + "=" * 80)
    print("WHAT PREDICTS V4 ERRORS?")
    print("=" * 80)
    
    # Create target: was V4 wrong?
    v4['v4_wrong'] = (v4['v4_correct'] == 0).astype(float)
    
    features = [
        'v4_confidence', 'quality_diff', 'rank_gap', 'is_danger_zone', 'is_mismatch',
        'vol3_sum', 'vol3_diff', 'to_pressure_diff', 'chaos_index', 'expected_tempo',
        'floor_diff', 'ceiling_diff', 'threep_matchup_adv', 'reb_matchup',
        'h_3p_pct', 'a_3p_pct', 'pace', 'total_points'
    ]
    
    print("\n--- Correlation with V4 Being Wrong ---\n")
    print(f"{'Feature':<25} {'Correlation':>15}")
    print("-" * 45)
    
    correlations = []
    for feat in features:
        if feat in v4.columns:
            corr = v4[feat].corr(v4['v4_wrong'])
            if not np.isnan(corr):
                correlations.append((feat, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feat, corr in correlations:
        print(f"{feat:<25} {corr:>+15.3f}")
    
    print("\nPositive = higher value means MORE errors")
    print("Negative = higher value means FEWER errors")

def main():
    print("=" * 80)
    print("V4 ERROR DEEP DIVE - FINDING THE BIGGEST OPPORTUNITY")
    print("=" * 80)
    
    games, teams = load_data()
    games = add_team_stats(games, teams)
    games = build_proposed_features(games)
    
    v4, wrong, right = analyze_v4_errors(games)
    test_proposed_features(v4, wrong, right)
    analyze_biggest_opportunities(v4, wrong)
    analyze_what_predicts_upsets(v4, wrong)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()