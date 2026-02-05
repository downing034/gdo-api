"""
Dual GA: Optimize for Home Score and Away Score Separately

Instead of predicting margin, predict actual scores.
This should improve both spread AND total predictions.

Runtime target: ~2 hours per target = ~4 hours total
"""

import os
import random
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
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

# Reduced parameters for ~2 hour runtime per target
POPULATION_SIZE = 200
GENERATIONS = 200
NUM_RESTARTS = 4
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.35
MUTATION_FLIP_RATE = 0.06
ELITE_SIZE = 5
MIN_FEATURES = 12
MAX_FEATURES = 40
N_JOBS = multiprocessing.cpu_count()
HALL_OF_FAME_SIZE = 500


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
        start_year = int('20' + season.split('_')[0])
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
    
    print(f"  Games: {len(df)}")
    return df


def generate_features_for_score(df, target_team='home'):
    """
    Generate features optimized for predicting a specific team's score.
    
    For home_score: focus on home offensive + away defensive features
    For away_score: focus on away offensive + home defensive features
    """
    print(f"\nGenerating features for {target_team}_score prediction...")
    features = {}
    
    # Determine offensive and defensive team
    off_team = target_team  # Team whose score we're predicting
    def_team = 'away' if target_team == 'home' else 'home'  # Opponent
    
    # ==========================================================================
    # OFFENSIVE TEAM FEATURES (their ability to score)
    # ==========================================================================
    
    # Rolling offensive stats
    for w in ['5', '10']:
        features[f'{off_team}_AdjO_{w}'] = df[f'{off_team}_AdjO_rolling_{w}'].values.astype(float)
        features[f'{off_team}_eFG_off_{w}'] = df[f'{off_team}_eFG_off_rolling_{w}'].values.astype(float)
        features[f'{off_team}_TOV_off_{w}'] = df[f'{off_team}_TOV_off_rolling_{w}'].values.astype(float)
        features[f'{off_team}_OReb_{w}'] = df[f'{off_team}_OReb_rolling_{w}'].values.astype(float)
        features[f'{off_team}_FTR_off_{w}'] = df[f'{off_team}_FTR_off_rolling_{w}'].values.astype(float)
        features[f'{off_team}_g_score_{w}'] = df[f'{off_team}_g_score_rolling_{w}'].values.astype(float)
        features[f'{off_team}_tempo_{w}'] = df[f'{off_team}_T_rolling_{w}'].values.astype(float)
    
    # Season offensive stats
    features[f'{off_team}_adj_off'] = df[f'{off_team}_adj_off'].values.astype(float)
    features[f'{off_team}_efg_off'] = df[f'{off_team}_efg_off'].values.astype(float)
    features[f'{off_team}_tov_off'] = df[f'{off_team}_tov_off'].values.astype(float)
    features[f'{off_team}_oreb'] = df[f'{off_team}_oreb'].values.astype(float)
    features[f'{off_team}_ftr_off'] = df[f'{off_team}_ftr_off'].values.astype(float)
    features[f'{off_team}_3p_off'] = df[f'{off_team}_3p_off'].values.astype(float)
    features[f'{off_team}_2p_off'] = df[f'{off_team}_2p_off'].values.astype(float)
    features[f'{off_team}_ft_pct'] = df[f'{off_team}_ft_pct'].values.astype(float)
    features[f'{off_team}_tempo'] = df[f'{off_team}_tempo'].values.astype(float)
    features[f'{off_team}_talent'] = df[f'{off_team}_talent'].values.astype(float)
    features[f'{off_team}_assist_off'] = df[f'{off_team}_assist_off'].values.astype(float)
    features[f'{off_team}_3p_rate_off'] = df[f'{off_team}_3p_rate_off'].values.astype(float)
    
    # Quality metrics
    features[f'{off_team}_weighted_quality'] = df[f'{off_team}_weighted_quality'].values.astype(float)
    features[f'{off_team}_quality_score'] = df[f'{off_team}_quality_score'].values.astype(float)
    features[f'{off_team}_barthag'] = df[f'{off_team}_barthag'].values.astype(float)
    features[f'{off_team}_off_rank'] = df[f'{off_team}_off_rank'].values.astype(float)
    
    # ==========================================================================
    # DEFENSIVE TEAM FEATURES (opponent's ability to stop scoring)
    # ==========================================================================
    
    # Rolling defensive stats
    for w in ['5', '10']:
        features[f'{def_team}_AdjD_{w}'] = df[f'{def_team}_AdjD_rolling_{w}'].values.astype(float)
        features[f'{def_team}_eFG_def_{w}'] = df[f'{def_team}_eFG_def_rolling_{w}'].values.astype(float)
        features[f'{def_team}_TOV_def_{w}'] = df[f'{def_team}_TOV_def_rolling_{w}'].values.astype(float)
        features[f'{def_team}_DReb_{w}'] = df[f'{def_team}_DReb_rolling_{w}'].values.astype(float)
        features[f'{def_team}_FTR_def_{w}'] = df[f'{def_team}_FTR_def_rolling_{w}'].values.astype(float)
    
    # Season defensive stats
    features[f'{def_team}_adj_def'] = df[f'{def_team}_adj_def'].values.astype(float)
    features[f'{def_team}_efg_def'] = df[f'{def_team}_efg_def'].values.astype(float)
    features[f'{def_team}_tov_def'] = df[f'{def_team}_tov_def'].values.astype(float)
    features[f'{def_team}_dreb'] = df[f'{def_team}_dreb'].values.astype(float)
    features[f'{def_team}_ftr_def'] = df[f'{def_team}_ftr_def'].values.astype(float)
    features[f'{def_team}_3p_def'] = df[f'{def_team}_3p_def'].values.astype(float)
    features[f'{def_team}_2p_def'] = df[f'{def_team}_2p_def'].values.astype(float)
    features[f'{def_team}_block_def'] = df[f'{def_team}_block_def'].values.astype(float)
    features[f'{def_team}_def_rank'] = df[f'{def_team}_def_rank'].values.astype(float)
    
    # ==========================================================================
    # MATCHUP FEATURES (offense vs defense)
    # ==========================================================================
    
    # Efficiency matchup
    for w in ['5', '10']:
        features[f'off_vs_def_eff_{w}'] = features[f'{off_team}_AdjO_{w}'] - features[f'{def_team}_AdjD_{w}']
    
    # eFG matchup
    for w in ['5', '10']:
        features[f'eFG_matchup_{w}'] = features[f'{off_team}_eFG_off_{w}'] - features[f'{def_team}_eFG_def_{w}']
    
    # TOV matchup (lower is better for offense)
    for w in ['5', '10']:
        features[f'TOV_matchup_{w}'] = features[f'{def_team}_TOV_def_{w}'] - features[f'{off_team}_TOV_off_{w}']
    
    # Rebounding matchup
    for w in ['5', '10']:
        features[f'reb_matchup_{w}'] = features[f'{off_team}_OReb_{w}'] - features[f'{def_team}_DReb_{w}']
    
    # FT matchup
    for w in ['5', '10']:
        features[f'FTR_matchup_{w}'] = features[f'{off_team}_FTR_off_{w}'] - features[f'{def_team}_FTR_def_{w}']
    
    # 3pt matchup
    features['3p_matchup'] = features[f'{off_team}_3p_off'] - features[f'{def_team}_3p_def']
    
    # Quality matchup
    features['quality_matchup'] = features[f'{off_team}_weighted_quality'] - df[f'{def_team}_weighted_quality'].values.astype(float)
    
    # ==========================================================================
    # TEMPO/PACE FEATURES
    # ==========================================================================
    
    features['avg_tempo_5'] = (df['home_T_rolling_5'] + df['away_T_rolling_5']).values.astype(float) / 2
    features['avg_tempo_10'] = (df['home_T_rolling_10'] + df['away_T_rolling_10']).values.astype(float) / 2
    features['avg_tempo_season'] = (df['home_tempo'] + df['away_tempo']).values.astype(float) / 2
    
    # ==========================================================================
    # CONTEXT FEATURES
    # ==========================================================================
    
    features['is_neutral'] = (df['venue'] == 'N').astype(float).values
    features['is_conference'] = (df['home_conf'] == df['away_conf']).astype(float).values
    features[f'{off_team}_days_rest'] = np.clip(df[f'{off_team}_days_rest'].values, 0, 14).astype(float)
    features[f'{def_team}_days_rest'] = np.clip(df[f'{def_team}_days_rest'].values, 0, 14).astype(float)
    features['rest_diff'] = features[f'{off_team}_days_rest'] - features[f'{def_team}_days_rest']
    
    # Home court matters more for home team scoring
    if target_team == 'home':
        features['home_court'] = np.where(features['is_neutral'] == 1, 0.0,
                                          np.where(features['is_conference'] == 1, 2.5, 3.5))
    else:
        features['home_court'] = np.where(features['is_neutral'] == 1, 0.0,
                                          np.where(features['is_conference'] == 1, -2.5, -3.5))
    
    # SOS
    features[f'{off_team}_sos'] = df[f'{off_team}_sos'].values.astype(float)
    features[f'{def_team}_sos'] = df[f'{def_team}_sos'].values.astype(float)
    
    # ==========================================================================
    # POLYNOMIAL FEATURES
    # ==========================================================================
    
    for f in ['off_vs_def_eff_5', 'off_vs_def_eff_10', 'quality_matchup', 'eFG_matchup_5']:
        if f in features:
            features[f'{f}_sq'] = features[f] ** 2
            features[f'{f}_sqrt'] = np.sign(features[f]) * np.sqrt(np.abs(features[f]))
    
    # ==========================================================================
    # INTERACTION FEATURES
    # ==========================================================================
    
    # Tempo interactions (high tempo = more scoring opportunities)
    features['off_eff_x_tempo'] = features[f'{off_team}_AdjO_5'] * features['avg_tempo_5'] / 70
    features['matchup_x_tempo'] = features['off_vs_def_eff_5'] * features['avg_tempo_5'] / 70
    
    # Context interactions
    features['matchup_x_home'] = features['off_vs_def_eff_5'] * features['home_court']
    features['matchup_x_conf'] = features['off_vs_def_eff_5'] * features['is_conference']
    features['matchup_x_neutral'] = features['off_vs_def_eff_5'] * features['is_neutral']
    
    # Quality x context
    features['quality_x_home'] = features['quality_matchup'] * features['home_court']
    
    print(f"  Generated {len(features)} features")
    return features


def filter_features(features, target, min_corr=0.02):
    """Filter and deduplicate features."""
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
        if len(kept) >= 300:
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
    individual, = args
    global _X_global, _y_global
    
    selected_idx = np.where(individual)[0]
    if len(selected_idx) < MIN_FEATURES:
        return -100.0, individual
    
    X_subset = _X_global[:, selected_idx]
    
    try:
        model = Ridge(alpha=1.0)
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
    def __init__(self, X, y, feature_names, target_name='score'):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.target_name = target_name
        
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
            
            if gen % 25 == 0:
                print(f"    Gen {gen}: best={-fitnesses[best_idx]:.3f}, global={-self.best_fitness:.3f}, unique={self.total_unique}")
            
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
        print(f"GA FOR {self.target_name.upper()}")
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
        print(f"Best MAE: {-self.best_fitness:.3f}")
        
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


def final_eval(X, y, feature_names, selected, target_name):
    """Final XGBoost evaluation."""
    print(f"\n--- Final XGBoost Eval for {target_name} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sel_idx = [feature_names.index(f) for f in selected if f in feature_names]
    
    params = {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 
              'verbosity': 0, 'n_jobs': -1}
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train[:, sel_idx], y_train)
    
    test_pred = model.predict(X_test[:, sel_idx])
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    print(f"  Features: {len(sel_idx)}")
    print(f"  Test MAE: {test_mae:.2f}")
    
    imp = pd.DataFrame({'f': [feature_names[i] for i in sel_idx], 
                       'i': model.feature_importances_}).sort_values('i', ascending=False)
    print("\n  Top 10 importances:")
    for _, row in imp.head(10).iterrows():
        print(f"    {row['f']:<40} {row['i']:.4f}")
    
    return {'mae': test_mae, 'features': selected, 'model': model}


def main():
    print("="*70)
    print("DUAL GA: HOME_SCORE + AWAY_SCORE")
    print("="*70)
    
    df = load_data()
    
    results = {}
    
    # ==================== HOME SCORE ====================
    print("\n" + "="*70)
    print("PHASE 1: HOME SCORE PREDICTION")
    print("="*70)
    
    home_target = df['home_score'].values.astype(float)
    home_features = generate_features_for_score(df, 'home')
    home_valid, home_corr = filter_features(home_features, home_target)
    
    sorted_corr = sorted(home_corr.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 home_score correlations:")
    for i, (n, c) in enumerate(sorted_corr[:10]):
        print(f"  {i+1}. {n:<40} r={c:.4f}")
    
    home_names = list(home_valid.keys())
    X_home = np.column_stack([home_valid[n] for n in home_names])
    X_home = np.nan_to_num(X_home, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_home_train, _, y_home_train, _ = train_test_split(X_home, home_target, test_size=0.2, random_state=42)
    
    ga_home = FastGA(X_home_train, y_home_train, home_names, 'home_score')
    ga_home.run()
    
    home_selected = ga_home.get_selected_features()
    print(f"\nHome Score Selected Features ({len(home_selected)}):")
    for f in home_selected:
        print(f"  - {f}")
    
    results['home'] = final_eval(X_home, home_target, home_names, home_selected, 'home_score')
    results['home']['hof_freq'] = ga_home.get_hof_frequency()
    
    # ==================== AWAY SCORE ====================
    print("\n" + "="*70)
    print("PHASE 2: AWAY SCORE PREDICTION")
    print("="*70)
    
    away_target = df['away_score'].values.astype(float)
    away_features = generate_features_for_score(df, 'away')
    away_valid, away_corr = filter_features(away_features, away_target)
    
    sorted_corr = sorted(away_corr.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 away_score correlations:")
    for i, (n, c) in enumerate(sorted_corr[:10]):
        print(f"  {i+1}. {n:<40} r={c:.4f}")
    
    away_names = list(away_valid.keys())
    X_away = np.column_stack([away_valid[n] for n in away_names])
    X_away = np.nan_to_num(X_away, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_away_train, _, y_away_train, _ = train_test_split(X_away, away_target, test_size=0.2, random_state=42)
    
    ga_away = FastGA(X_away_train, y_away_train, away_names, 'away_score')
    ga_away.run()
    
    away_selected = ga_away.get_selected_features()
    print(f"\nAway Score Selected Features ({len(away_selected)}):")
    for f in away_selected:
        print(f"  - {f}")
    
    results['away'] = final_eval(X_away, away_target, away_names, away_selected, 'away_score')
    results['away']['hof_freq'] = ga_away.get_hof_frequency()
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Home Score MAE: {results['home']['mae']:.2f}")
    print(f"Away Score MAE: {results['away']['mae']:.2f}")
    print(f"Combined (avg): {(results['home']['mae'] + results['away']['mae'])/2:.2f}")
    
    # Save results
    output = {
        'home_features': results['home']['features'],
        'away_features': results['away']['features'],
        'home_mae': results['home']['mae'],
        'away_mae': results['away']['mae'],
        'home_hof_top20': results['home']['hof_freq'][:20],
        'away_hof_top20': results['away']['hof_freq'][:20],
    }
    
    output_path = os.path.join(SCRIPT_DIR, 'dual_ga_results.txt')
    with open(output_path, 'w') as f:
        f.write("# Dual GA Results\n\n")
        f.write(f"# Home Score MAE: {results['home']['mae']:.2f}\n")
        f.write(f"home_features = {results['home']['features']}\n\n")
        f.write(f"# Away Score MAE: {results['away']['mae']:.2f}\n")
        f.write(f"away_features = {results['away']['features']}\n\n")
        f.write("# Home HOF Frequency:\n")
        for name, count in results['home']['hof_freq'][:20]:
            f.write(f"#   {name}: {count}\n")
        f.write("\n# Away HOF Frequency:\n")
        for name, count in results['away']['hof_freq'][:20]:
            f.write(f"#   {name}: {count}\n")
    
    print(f"\nSaved to: {output_path}")
    
    print("\n" + "="*70)
    print("FEATURE LISTS FOR V3")
    print("="*70)
    print(f"\nhome_features = {results['home']['features']}")
    print(f"\naway_features = {results['away']['features']}")


if __name__ == '__main__':
    main()