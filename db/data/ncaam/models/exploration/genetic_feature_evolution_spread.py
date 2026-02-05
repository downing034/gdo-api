"""
FAST Genetic Algorithm - Actually completes in reasonable time

Optimizations:
1. Ridge regression for fitness (100x faster than XGBoost)
2. Parallel evaluation with all cores
3. Smaller population, more generations
4. Final XGBoost evaluation only on top solutions
"""

import os
import random
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy import stats
import warnings
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
GAME_DATA_PATH = os.path.join(BASE_DIR, 'processed', 'base_model_game_data_with_rolling.csv')
TEAM_DATA_PATH = os.path.join(BASE_DIR, 'processed', 'ncaam_team_data_final.csv')

# Fast parameters
POPULATION_SIZE = 200
GENERATIONS = 300
NUM_RESTARTS = 10
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.35
MUTATION_FLIP_RATE = 0.06
ELITE_SIZE = 5
MIN_FEATURES = 12
MAX_FEATURES = 40
N_JOBS = multiprocessing.cpu_count()

HALL_OF_FAME_SIZE = 500

print(f"FAST GA MODE - {N_JOBS} cores")


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


def load_data():
    print("\nLoading data...")
    df = pd.read_csv(GAME_DATA_PATH, keep_default_na=False, na_values=[''])
    df['date'] = pd.to_datetime(df['date'])
    
    def get_cutoff_date(row):
        season = row['season']
        if pd.isna(season):
            return pd.Timestamp('1900-01-01')
        start_year = int('20' + str(season).split('_')[0])
        return pd.Timestamp(f'{start_year}-11-20')
    
    df['cutoff_date'] = df.apply(get_cutoff_date, axis=1)
    df = df[df['date'] >= df['cutoff_date']]
    
    teams = pd.read_csv(TEAM_DATA_PATH, keep_default_na=False, na_values=[''])
    
    team_cols = [
        ('Team ID', 'barthag_rank', 180), ('Adj. Off. Eff', 'adj_off', 100),
        ('Adj. Off. Eff Rank', 'off_rank', 180), ('Adj. Def. Eff', 'adj_def', 100),
        ('Adj. Def. Eff Rank', 'def_rank', 180), ('Barthag', 'barthag', 0.5),
        ('Eff. FG% Off', 'efg_off', 50), ('Eff. FG% Def', 'efg_def', 50),
        ('FT Rate Off', 'ftr_off', 30), ('FT Rate Def', 'ftr_def', 30),
        ('Turnover% Off', 'tov_off', 18), ('Turnover% Def', 'tov_def', 18),
        ('Off. Reb%', 'oreb', 30), ('Def. Reb%', 'dreb', 70),
        ('3P% Off', '3p_off', 33), ('3P% Def', '3p_def', 33),
        ('2P% Off', '2p_off', 50), ('2P% Def', '2p_def', 50),
        ('FT% Off', 'ft_pct', 70), ('Adj. Tempo', 'tempo', 68),
        ('Experience', 'experience', 2), ('Talent', 'talent', 0),
        ('Elite SOS', 'elite_sos', 0), ('Block% Def', 'block_def', 10),
        ('Assist% Off', 'assist_off', 50), ('3P Rate Off', '3p_rate_off', 35),
        ('Avg Height', 'height', 75), ('Q1 Wins', 'q1_wins', 0),
        ('Q1 Losses', 'q1_losses', 0), ('Q2 Wins', 'q2_wins', 0),
        ('Q2 Losses', 'q2_losses', 0), ('Quality Score', 'quality_score', 0),
        ('Weighted Quality', 'weighted_quality', 0),
    ]
    
    team_lookup = {}
    for _, row in teams.iterrows():
        code = row.get('Team_Code')
        if code:
            team_lookup[code] = {col[1]: safe_float(row.get(col[0]), col[2]) for col in team_cols}
    
    for prefix, col in [('home', 'home_team'), ('away', 'away_team')]:
        for _, key, default in team_cols:
            df[f'{prefix}_{key}'] = df[col].apply(lambda x: team_lookup.get(x, {}).get(key, default))
    
    df['margin'] = df['home_score'] - df['away_score']
    print(f"  Games: {len(df)}")
    return df


def generate_features(df):
    print("\nGenerating features...")
    features = {}
    
    # Raw rolling stats
    for prefix in ['home', 'away']:
        for stat in ['AdjO', 'AdjD', 'T', 'eFG_off', 'eFG_def', 'TOV_off', 'TOV_def',
                     'OReb', 'DReb', 'FTR_off', 'FTR_def', 'g_score']:
            for w in ['5', '10']:
                col = f'{prefix}_{stat}_rolling_{w}'
                if col in df.columns:
                    features[f'{prefix}_{stat}_{w}'] = df[col].values.astype(float)
        
        for stat in ['adj_off', 'adj_def', 'off_rank', 'def_rank', 'barthag_rank', 'barthag',
                     'efg_off', 'efg_def', 'ftr_off', 'ftr_def', 'tov_off', 'tov_def',
                     'oreb', 'dreb', '3p_off', '3p_def', '2p_off', '2p_def', 'ft_pct',
                     'tempo', 'experience', 'talent', 'elite_sos', 'block_def', 'assist_off',
                     '3p_rate_off', 'height', 'q1_wins', 'q1_losses', 'q2_wins', 'q2_losses',
                     'quality_score', 'weighted_quality']:
            col = f'{prefix}_{stat}'
            if col in df.columns:
                features[col] = df[col].values.astype(float)
        
        features[f'{prefix}_sos'] = df[f'{prefix}_sos'].values.astype(float)
        features[f'{prefix}_days_rest'] = np.clip(df[f'{prefix}_days_rest'].values, 0, 14).astype(float)
    
    features['is_neutral'] = (df['venue'] == 'N').astype(float).values
    features['is_conference'] = (df['home_conf'] == df['away_conf']).astype(float).values
    features['rest_diff'] = np.clip(df['home_days_rest'] - df['away_days_rest'], -10, 10).astype(float)
    
    # Efficiency margins
    for w in ['5', '10']:
        if f'home_AdjO_{w}' in features:
            features[f'home_eff_margin_{w}'] = features[f'home_AdjO_{w}'] - features[f'home_AdjD_{w}']
            features[f'away_eff_margin_{w}'] = features[f'away_AdjO_{w}'] - features[f'away_AdjD_{w}']
            features[f'eff_margin_diff_{w}'] = features[f'home_eff_margin_{w}'] - features[f'away_eff_margin_{w}']
    
    # All diffs
    for key in list(features.keys()):
        if key.startswith('home_'):
            away_key = 'away_' + key[5:]
            if away_key in features:
                diff_key = key[5:] + '_diff'
                if diff_key not in features:
                    features[diff_key] = features[key] - features[away_key]
    
    # Home court
    features['home_court'] = np.where(features['is_neutral'] == 1, 0.0,
                                       np.where(features['is_conference'] == 1, 2.5, 3.5))
    
    # Matchups
    for off_stat, def_stat in [('eFG_off', 'eFG_def'), ('TOV_off', 'TOV_def'), ('3p_off', '3p_def')]:
        for w in ['5', '']:
            suffix = f'_{w}' if w else ''
            ho, ad = f'home_{off_stat}{suffix}', f'away_{def_stat}{suffix}'
            ao, hd = f'away_{off_stat}{suffix}', f'home_{def_stat}{suffix}'
            if all(k in features for k in [ho, ad, ao, hd]):
                features[f'{off_stat}_matchup_diff{suffix}'] = (features[ho] - features[ad]) - (features[ao] - features[hd])
    
    # Polynomials
    for f in ['eff_margin_diff_5', 'eff_margin_diff_10', 'barthag_rank_diff', 
              'def_rank_diff', 'barthag_diff', 'sos_diff', 'talent_diff']:
        if f in features:
            arr = features[f]
            features[f'{f}_sq'] = arr ** 2
            features[f'{f}_signed_sq'] = np.sign(arr) * (arr ** 2)
            features[f'{f}_abs'] = np.abs(arr)
            features[f'{f}_sqrt'] = np.sign(arr) * np.sqrt(np.abs(arr))
    
    # Context interactions
    for feat in ['eff_margin_diff_5', 'eff_margin_diff_10', 'barthag_rank_diff', 
                 'barthag_diff', 'def_rank_diff', 'sos_diff', 'talent_diff',
                 'eFG_off_diff_5', 'TOV_off_diff_5', 'g_score_5_diff']:
        for ctx in ['home_court', 'is_neutral', 'is_conference', 'rest_diff']:
            if feat in features and ctx in features:
                features[f'{feat}_x_{ctx}'] = features[feat] * features[ctx]
    
    # Feature x feature
    for f1, f2 in combinations(['eff_margin_diff_5', 'barthag_rank_diff', 'barthag_diff', 
                                 'def_rank_diff', 'sos_diff', 'eFG_off_diff_5', 'talent_diff'], 2):
        if f1 in features and f2 in features:
            features[f'{f1}_x_{f2}'] = features[f1] * features[f2]
    
    # Ratios
    if 'home_eff_margin_5' in features and 'home_barthag_rank' in features:
        features['home_eff_per_rank'] = safe_divide(features['home_eff_margin_5'], features['home_barthag_rank']) * 100
        features['away_eff_per_rank'] = safe_divide(features['away_eff_margin_5'], features['away_barthag_rank']) * 100
        features['eff_per_rank_diff'] = features['home_eff_per_rank'] - features['away_eff_per_rank']
    
    # Composites
    if all(k in features for k in ['eff_margin_diff_5', 'barthag_diff', 'sos_diff']):
        features['composite_power'] = features['eff_margin_diff_5'] * 0.5 + features['barthag_diff'] * 10 + features['sos_diff'] * 0.3
    
    if all(k in features for k in ['eFG_off_diff_5', 'TOV_off_diff_5', 'OReb_diff_5', 'FTR_off_diff_5']):
        features['composite_four_factors'] = (features['eFG_off_diff_5'] * 0.4 + features['TOV_off_diff_5'] * 0.25 +
                                               features['OReb_diff_5'] * 0.2 + features['FTR_off_diff_5'] * 0.15)
    
    # Tiers
    if 'home_barthag_rank' in features:
        def to_tier(r):
            t = np.ones_like(r) * 5
            t = np.where(r <= 25, 1, t)
            t = np.where((r > 25) & (r <= 75), 2, t)
            t = np.where((r > 75) & (r <= 150), 3, t)
            t = np.where((r > 150) & (r <= 250), 4, t)
            return t
        features['tier_diff'] = to_tier(features['away_barthag_rank']) - to_tier(features['home_barthag_rank'])
    
    print(f"  Total: {len(features)}")
    return features


def filter_features(features, target, min_corr=0.01):
    print("\nFiltering...")
    valid = {}
    correlations = {}
    
    for name, values in features.items():
        values = np.array(values, dtype=float)
        finite_mask = np.isfinite(values)
        if finite_mask.sum() < len(target) * 0.8:
            continue
        median_val = np.nanmedian(values[finite_mask]) if finite_mask.any() else 0
        values = np.where(finite_mask, values, median_val)
        try:
            r, _ = stats.pearsonr(values, target)
            if np.isfinite(r) and abs(r) >= min_corr:
                valid[name] = values
                correlations[name] = abs(r)
        except:
            continue
    
    # Remove highly correlated
    sorted_feats = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)
    kept = []
    kept_vals = {}
    for name in sorted_feats:
        dominated = False
        for kn in kept:
            try:
                r, _ = stats.pearsonr(valid[name], kept_vals[kn])
                if abs(r) > 0.98:
                    dominated = True
                    break
            except:
                continue
        if not dominated:
            kept.append(name)
            kept_vals[name] = valid[name]
        if len(kept) >= 400:
            break
    
    print(f"  Kept: {len(kept)}")
    return {n: valid[n] for n in kept}, {n: correlations[n] for n in kept}


# Global variables for parallel evaluation
_X_global = None
_y_global = None


def init_worker(X, y):
    global _X_global, _y_global
    _X_global = X
    _y_global = y


def evaluate_individual(args):
    """Fast evaluation with Ridge regression."""
    individual, = args
    global _X_global, _y_global
    
    selected_idx = np.where(individual)[0]
    if len(selected_idx) < MIN_FEATURES:
        return -100.0, individual
    
    X_subset = _X_global[:, selected_idx]
    
    try:
        model = Ridge(alpha=1.0)
        # Single train/val split for speed
        n = len(_y_global)
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[:int(n*0.8)], idx[int(n*0.8):]
        
        model.fit(X_subset[train_idx], _y_global[train_idx])
        pred = model.predict(X_subset[val_idx])
        mae = np.mean(np.abs(pred - _y_global[val_idx]))
        return -mae, individual
    except:
        return -100.0, individual


class FastGA:
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        self.tested_hashes = set()
        self.hall_of_fame = []
        self.total_unique = 0
        self.best_individual = None
        self.best_fitness = -np.inf
    
    def create_individual(self):
        n_selected = random.randint(MIN_FEATURES, MAX_FEATURES)
        selected = random.sample(range(self.n_features), n_selected)
        mask = np.zeros(self.n_features, dtype=bool)
        mask[selected] = True
        return mask
    
    def individual_hash(self, ind):
        return hashlib.md5(ind.tobytes()).hexdigest()
    
    def evaluate_population(self, population):
        """Parallel evaluation."""
        args_list = [(ind,) for ind in population]
        
        with ProcessPoolExecutor(max_workers=N_JOBS, 
                                  initializer=init_worker, 
                                  initargs=(self.X, self.y)) as executor:
            results = list(executor.map(evaluate_individual, args_list))
        
        fitnesses = []
        for fitness, ind in results:
            h = self.individual_hash(ind)
            if h not in self.tested_hashes:
                self.tested_hashes.add(h)
                self.total_unique += 1
            
            fitnesses.append(fitness)
            
            # Hall of fame
            if len(self.hall_of_fame) < HALL_OF_FAME_SIZE:
                self.hall_of_fame.append((fitness, ind.copy()))
                self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
            elif fitness > self.hall_of_fame[-1][0]:
                self.hall_of_fame[-1] = (fitness, ind.copy())
                self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        
        return fitnesses
    
    def tournament_select(self, population, fitnesses):
        indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()
    
    def crossover(self, p1, p2):
        if random.random() > CROSSOVER_RATE:
            return p1.copy(), p2.copy()
        mask = np.random.random(self.n_features) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)
    
    def mutate(self, ind):
        for i in range(self.n_features):
            if random.random() < MUTATION_FLIP_RATE:
                ind[i] = not ind[i]
        
        if random.random() < 0.2:
            n_change = random.randint(2, 6)
            selected = np.where(ind)[0]
            if len(selected) > MIN_FEATURES + n_change:
                to_remove = random.sample(list(selected), n_change)
                ind[to_remove] = False
            unselected = np.where(~ind)[0]
            if len(unselected) >= n_change:
                to_add = random.sample(list(unselected), n_change)
                ind[to_add] = True
        
        n_sel = ind.sum()
        if n_sel < MIN_FEATURES:
            unsel = np.where(~ind)[0]
            ind[random.sample(list(unsel), MIN_FEATURES - n_sel)] = True
        elif n_sel > MAX_FEATURES:
            sel = np.where(ind)[0]
            ind[random.sample(list(sel), n_sel - MAX_FEATURES)] = False
        
        return ind
    
    def run_single(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        population = [self.create_individual() for _ in range(POPULATION_SIZE)]
        
        for gen in range(GENERATIONS):
            fitnesses = self.evaluate_population(population)
            
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = population[best_idx].copy()
            
            if gen % 30 == 0:
                print(f"    Gen {gen}: best={-fitnesses[best_idx]:.4f}, global={-self.best_fitness:.4f}, unique={self.total_unique}")
            
            sorted_idx = np.argsort(fitnesses)[::-1]
            next_pop = [population[i].copy() for i in sorted_idx[:ELITE_SIZE]]
            
            while len(next_pop) < POPULATION_SIZE:
                p1 = self.tournament_select(population, fitnesses)
                p2 = self.tournament_select(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                next_pop.extend([self.mutate(c1), self.mutate(c2)])
            
            population = next_pop[:POPULATION_SIZE]
    
    def run(self):
        print(f"\n{'='*70}")
        print("FAST GA")
        print(f"{'='*70}")
        print(f"Restarts: {NUM_RESTARTS}, Pop: {POPULATION_SIZE}, Gen: {GENERATIONS}")
        
        start = time.time()
        
        for r in range(NUM_RESTARTS):
            print(f"\n--- Restart {r+1}/{NUM_RESTARTS} ---")
            self.run_single(r * 1000 + 42)
            print(f"  Unique so far: {self.total_unique}")
        
        print(f"\n{'='*70}")
        print(f"Time: {time.time()-start:.1f}s")
        print(f"Unique tested: {self.total_unique}")
        print(f"Best MAE: {-self.best_fitness:.4f}")
        
        return self.best_individual
    
    def get_selected_features(self):
        if self.best_individual is None:
            return []
        return [self.feature_names[i] for i in np.where(self.best_individual)[0]]
    
    def get_hof_frequency(self, top_n=100):
        freq = {}
        for fit, ind in self.hall_of_fame[:top_n]:
            for i in np.where(ind)[0]:
                name = self.feature_names[i]
                freq[name] = freq.get(name, 0) + 1
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)


def final_eval(X, y, feature_names, selected, v1_features):
    print(f"\n{'='*70}")
    print("FINAL XGBOOST EVALUATION")
    print(f"{'='*70}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sel_idx = [feature_names.index(f) for f in selected if f in feature_names]
    v1_idx = [feature_names.index(f) for f in v1_features if f in feature_names]
    
    params = {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 
              'verbosity': 0, 'n_jobs': -1}
    
    results = {}
    for name, idx in [('V1', v1_idx), ('GA', sel_idx)]:
        if not idx:
            continue
        model = xgb.XGBRegressor(**params)
        model.fit(X_train[:, idx], y_train)
        
        test_pred = model.predict(X_test[:, idx])
        test_mae = np.mean(np.abs(test_pred - y_test))
        test_acc = np.mean((test_pred > 0) == (y_test > 0))
        
        print(f"\n{name} ({len(idx)} features): MAE={test_mae:.3f}, Acc={test_acc:.1%}")
        results[name] = {'mae': test_mae, 'acc': test_acc, 'model': model}
        
        if name == 'GA':
            imp = pd.DataFrame({'f': [feature_names[i] for i in idx], 
                               'i': model.feature_importances_}).sort_values('i', ascending=False)
            print("\nTop 15 importances:")
            for _, row in imp.head(15).iterrows():
                print(f"  {row['f']:<45} {row['i']:.4f}")
    
    if 'V1' in results and 'GA' in results:
        print(f"\nMAE: {results['GA']['mae'] - results['V1']['mae']:+.3f}")
        print(f"Acc: {results['GA']['acc'] - results['V1']['acc']:+.2%}")
    
    return results


def main():
    print("="*70)
    print("FAST GENETIC ALGORITHM")
    print("="*70)
    
    df = load_data()
    target = df['margin'].values
    features = generate_features(df)
    valid_features, correlations = filter_features(features, target)
    
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 by correlation:")
    for i, (n, c) in enumerate(sorted_corr[:10]):
        print(f"  {i+1}. {n:<45} r={c:.4f}")
    
    feature_names = list(valid_features.keys())
    X = np.column_stack([valid_features[n] for n in feature_names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train)}, Features: {len(feature_names)}")
    
    ga = FastGA(X_train, y_train, feature_names)
    ga.run()
    
    selected = ga.get_selected_features()
    print(f"\n{'='*70}")
    print(f"SELECTED ({len(selected)} features)")
    print(f"{'='*70}")
    for i, n in enumerate(selected):
        print(f"  {i+1}. {n}")
    
    print(f"\n{'='*70}")
    print("HOF FREQUENCY (top 30)")
    print(f"{'='*70}")
    for n, c in ga.get_hof_frequency()[:30]:
        print(f"  {n:<45} {c}")
    
    v1 = ['eff_margin_diff_5', 'eff_margin_diff_10', 'eFG_off_diff_5', 'eFG_def_diff_5',
          'TOV_off_diff_5', 'TOV_def_diff_5', 'OReb_diff_5', 'DReb_diff_5',
          'FTR_off_diff_5', 'FTR_def_diff_5', 'home_court', 'is_neutral', 'is_conference',
          'rest_diff', 'sos_diff', 'g_score_5_diff', 'g_score_10_diff']
    
    final_eval(X, target, feature_names, selected, v1)
    
    print(f"\n\nselected_features = {selected}")


if __name__ == '__main__':
    main()