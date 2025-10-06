import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import pytensor.tensor as pt
from scipy.stats import norm
from sqlalchemy import create_engine
from datetime import datetime

# Database connection details for SQLite
db_connection_str = 'sqlite:///f1db.db' # SQLite database file is f1db.db in the same directory
db_engine = create_engine(db_connection_str)
db_connection_results = 'sqlite:///f1_results.db' # Use a NEW DB for results
db_engine_results = create_engine(db_connection_results)

def fetch_data_from_sql():
    """
    Fetches and combines data from race_data, race, circuit, driver, and constructor
    tables, focusing on RACE_RESULT entries and structuring data for the model.
    """
    print("--- Fetching Data from SQL ---")

    results_query = """
    SELECT
        rd.race_id AS raceid,
        rd.position_display_order AS resultid, 
        rd.driver_id AS driverid,
        rd.constructor_id AS teamid,
        r.year AS year,
        rd.position_display_order AS position,
        CASE
            WHEN rd.race_reason_retired IS NOT NULL AND rd.race_reason_retired != '' THEN 1
            ELSE 0
        END AS is_dnf,
        rd.starting_grid_position_qualification_position_number AS grid,
        r.circuit_id AS circuitid,
        c.type AS circuit_type,
        rd.position_text AS position_description,
        rd.race_reason_retired AS dnf_reason
        -- TODO: Add a column here if your DB can indicate customer car entries, e.g.
        -- , entrant.is_customer_car AS is_customer_car_raw 
        -- This would require joining with an 'entrant' table or similar
    FROM race_data rd
    JOIN race r ON rd.race_id = r.id
    JOIN circuit c ON r.circuit_id = c.id
    WHERE rd."type" = 'RACE_RESULT'
      AND r.grand_prix_id NOT IN (SELECT id FROM grand_prix WHERE full_name LIKE '%Indianapolis 500%')
    ORDER BY
        rd.race_id,
        rd.position_display_order;
    """
    try:
        results_df = pd.read_sql_query(results_query, db_engine)
        print(f"Fetched {len(results_df)} rows from race_data (type='RACE_RESULT') joined with race and circuit.")
    except Exception as e:
        print(f"!!! Error executing results_query: {e}")
        print("Query was:\n", results_query)
        return None, None, None

    if results_df.empty:
        print("!!! Warning: results_df is empty after query. Check DB contents and filters.")
        return None, None, None

    duplicates = results_df.duplicated(subset=['raceid', 'driverid'], keep=False)
    if duplicates.sum() > 0:
        print(f"!!! Warning: Found {duplicates.sum()} duplicate raceid/driverid entries.")
    
    if 'position' in results_df.columns:
        results_df.dropna(subset=['position'], inplace=True)
        results_df['position'] = pd.to_numeric(results_df['position'], errors='coerce').astype('Int64')
        if results_df['position'].isnull().any():
            print("!!! Warning: NULL values in 'position' after numeric conversion. Dropping these rows.")
            results_df.dropna(subset=['position'], inplace=True)
            results_df['position'] = results_df['position'].astype(int) 
    else:
        print("!!! Critical Warning: 'position' column missing from results_df query.")
        return None, None, None

    results_df['grid'] = pd.to_numeric(results_df['grid'], errors='coerce').astype('Int64')
    results_df['is_dnf'] = results_df['is_dnf'].astype(int)

    drivers_query = "SELECT id AS driverid, first_name AS forename, last_name AS surname, date_of_birth AS dob FROM driver"
    constructors_query = "SELECT id AS constructorid, name FROM constructor"
    try:
        drivers_df = pd.read_sql_query(drivers_query, db_engine)
        constructors_df = pd.read_sql_query(constructors_query, db_engine)
        print(f"Fetched {len(drivers_df)} drivers (with DOB) and {len(constructors_df)} constructors.")
    except Exception as e:
        print(f"!!! Error fetching driver/constructor data: {e}")
        drivers_df = pd.DataFrame(columns=['driverid', 'forename', 'surname', 'dob'])
        constructors_df = pd.DataFrame(columns=['constructorid', 'name'])

    if 'dob' in drivers_df.columns:
        results_df = pd.merge(results_df, drivers_df[['driverid', 'dob']], on='driverid', how='left')
    else:
        print("!!! Warning: 'dob' column not found in drivers_df. Age calculation will fail.")
        results_df['dob'] = pd.NaT

    if 'is_customer_car_raw' not in results_df.columns:
        print("Info: 'is_customer_car_raw' column not found. Initializing to 0. Populate this for customer car effect.")
        results_df['is_customer_car_raw'] = 0

    print("\nSample of processed results_df (Head):")
    print(results_df.head())
    results_df.info()
    print("--- Finished Fetching Data ---")
    return results_df, drivers_df, constructors_df

def calculate_points_robust(position):
    if pd.isna(position): return 0
    pos_int = int(position)
    if pos_int == 1: return 25
    elif pos_int == 2: return 18
    elif pos_int == 3: return 15
    elif pos_int == 4: return 12
    elif pos_int == 5: return 10
    elif pos_int == 6: return 8
    elif pos_int == 7: return 6
    elif pos_int == 8: return 4
    elif pos_int == 9: return 2
    elif pos_int == 10: return 1
    elif pos_int > 10: return 1.0 * (0.1)**((pos_int - 10) / 7.0)
    else: return 0

def rankit(data):
    n = len(data)
    if n == 0: return np.array([])
    if n == 1: return np.zeros(1)
    ranks = pd.Series(data).rank()
    epsilon = 1e-5 
    scaled_ranks = (ranks - 0.5) / n
    scaled_ranks = np.clip(scaled_ranks, epsilon, 1 - epsilon) 
    return norm.ppf(scaled_ranks)

def robustify_points_season(season_data):
    if len(season_data) < 2: return season_data
    df_copy = season_data.copy()
    df_copy['points'] = df_copy['points'].astype(float)
    min_points_val = df_copy['points'].min()
    max_points_val = df_copy['points'].max()
    if min_points_val == max_points_val: return df_copy 
    idx_worst_in_group = df_copy['points'].idxmin()
    idx_best_in_group = df_copy['points'].idxmax()
    df_copy.loc[idx_worst_in_group, 'points'] *= 0.5
    df_copy.loc[idx_best_in_group, 'points'] *= 0.5
    return df_copy

def prepare_data(race_results_df, drivers_df, constructors_df):
    print("--- Preparing Data for Model ---")
    if race_results_df is None or race_results_df.empty:
        print("!!! Error: Input race_results_df is None or empty.")
        return None
    data = race_results_df.copy()

    if 'dob' in data.columns:
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        data['race_date_approx'] = pd.to_datetime(data['year'].astype(str) + '-07-01', errors='coerce')
        valid_dates = data['dob'].notna() & data['race_date_approx'].notna()
        data.loc[valid_dates, 'age'] = ((data.loc[valid_dates, 'race_date_approx'] - data.loc[valid_dates, 'dob']).dt.days / 365.25)
        data['age'] = data['age'].round().astype('Int64')
        if data['age'].isnull().any():
            print(f"Warning: {data['age'].isnull().sum()} entries have missing age.")
    else:
        data['age'] = np.nan

    print("Calculating F1-specific experience (1950 onwards)...")
    f1_participation_set = set()
    for _, row in data[data['year'] >= 1950][['driverid', 'year']].drop_duplicates().iterrows():
        f1_participation_set.add((row['driverid'], row['year']))
    exp_list = []
    for driver_id_current, year_current in data[['driverid', 'year']].drop_duplicates().itertuples(index=False, name=None):
        current_exp = 0
        if year_current >= 1950: 
            for i in range(1, 5):
                lookback_year = year_current - i
                if lookback_year >= 1950 and (driver_id_current, lookback_year) in f1_participation_set:
                    current_exp += 1
        exp_list.append({'driverid': driver_id_current, 'year': year_current, 'experience_val': current_exp})
    experience_df = pd.DataFrame(exp_list)
    if 'experience' in data.columns: data = data.drop(columns=['experience'])
    data = pd.merge(data, experience_df.rename(columns={'experience_val': 'experience'}), on=['driverid', 'year'], how='left')
    data['experience'] = data['experience'].fillna(0).astype(int)

    data['is_customer_car_effect_active'] = data.apply(
        lambda row: 1 if (1950 <= row['year'] <= 1980 and row['is_customer_car_raw'] == 1) else 0, axis=1)
    data['points'] = data['position'].apply(calculate_points_robust)
    data = data.groupby(['driverid', 'year'], group_keys=False).apply(robustify_points_season)
    if data['points'].isnull().any() or np.isinf(data['points']).any():
        data['points'] = data['points'].replace([np.inf, -np.inf], np.nan).fillna(0)
    data['rankit_points'] = data.groupby('raceid')['points'].transform(lambda x: rankit(x.values))
    if data['rankit_points'].isnull().any() or np.isinf(data['rankit_points']).any():
        data['rankit_points'] = data['rankit_points'].replace([np.inf, -np.inf], np.nan).fillna(0)

    data['num_drivers_in_race'] = data.groupby('raceid')['driverid'].transform('count')
    data['norm_finish_pos'] = np.where(data['num_drivers_in_race'] > 0, data['position'] / data['num_drivers_in_race'], np.nan)
    driver_avg_norm_pos = data.groupby('driverid')['norm_finish_pos'].mean().rename('driver_avg_norm_pos')
    data = pd.merge(data, driver_avg_norm_pos, on='driverid', how='left')
    race_competitiveness = data.groupby('raceid')['driver_avg_norm_pos'].mean().rename('race_competitiveness')
    data = pd.merge(data, race_competitiveness, on='raceid', how='left')
    data['team_year'] = data['teamid'].astype(str) + '_' + data['year'].astype(str)

    age_knots = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47]
    age_bin_edges = [-np.inf] + age_knots + [np.inf]
    data['age_numeric'] = pd.to_numeric(data['age'], errors='coerce')
    data['age_bin_idx'] = pd.cut(data['age_numeric'], bins=age_bin_edges, labels=False, right=False)
    if data['age_bin_idx'].isnull().any():
        print(f"Warning: {data['age_bin_idx'].isnull().sum()} entries have NaN 'age_bin_idx'.")

    essential_cols = ['rankit_points', 'driverid', 'teamid', 'team_year',
                      'num_drivers_in_race', 'race_competitiveness', 'year',
                      'age_bin_idx', 'experience', 'is_customer_car_effect_active']
    initial_rows = len(data)
    data = data.dropna(subset=essential_cols)
    data['age_bin_idx'] = data['age_bin_idx'].astype(int)
    print(f"Dropped {initial_rows - len(data)} rows due to NaNs in essential columns.")
    if data.empty:
        print("!!! Error: DataFrame is empty after preparation.")
        return None
    print("--- Finished Preparing Data ---")
    return data

def create_model_year_variance_function(data):
    driver_ids = data['driverid'].unique()
    team_ids = data['teamid'].unique()
    team_year_ids = data['team_year'].unique()
    
    expected_age_bins = 11 
    num_experience_levels = 5 

    driver_idx = pd.Categorical(data['driverid'], categories=driver_ids).codes
    team_idx = pd.Categorical(data['teamid'], categories=team_ids).codes
    team_year_idx = pd.Categorical(data['team_year'], categories=team_year_ids).codes
    
    age_bin_indices = data['age_bin_idx'].values
    experience_indices = data['experience'].values
    is_customer_car_flags = data['is_customer_car_effect_active'].values
    num_drivers_in_race = data['num_drivers_in_race'].values
    race_competitiveness = data['race_competitiveness'].values
    rankit_points = data['rankit_points'].values
    year_centered = data['year'].values - data['year'].mean()

    with pm.Model() as model_year_vf_extended:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=1)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=1)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=1)
        beta_3 = pm.Normal('beta_3', mu=0, sigma=1)
        age_binned_coeffs = pm.Normal('age_binned_coeffs', mu=0, sigma=0.5, shape=expected_age_bins)
        experience_coeffs = pm.Normal('experience_coeffs', mu=0, sigma=0.5, shape=num_experience_levels)
        beta_customer = pm.Normal('beta_customer', mu=0, sigma=0.5)
        sigma_e = pm.HalfNormal('sigma_e', sigma=1)

        sd_dist_params = pm.HalfNormal.dist(sigma=1)
        chol_corr_u, _, sigmas_u = pm.LKJCholeskyCov('u_chol_corr_sigmas', n=2, eta=2.0, sd_dist=sd_dist_params, compute_corr=True)
        L_cov_u = pm.math.dot(pt.diag(sigmas_u), chol_corr_u)
        u_driver_params = pm.MvNormal('u_driver_params', mu=np.zeros(2), chol=L_cov_u, shape=(len(driver_ids), 2))

        chol_corr_v, _, sigmas_v = pm.LKJCholeskyCov('v_chol_corr_sigmas', n=2, eta=2.0, sd_dist=sd_dist_params, compute_corr=True)
        L_cov_v = pm.math.dot(pt.diag(sigmas_v), chol_corr_v)
        v_team_params = pm.MvNormal('v_team_params', mu=np.zeros(2), chol=L_cov_v, shape=(len(team_ids), 2))

        chol_corr_w, _, sigmas_w = pm.LKJCholeskyCov('w_chol_corr_sigmas', n=2, eta=2.0, sd_dist=sd_dist_params, compute_corr=True)
        L_cov_w = pm.math.dot(pt.diag(sigmas_w), chol_corr_w)
        w_team_year_params = pm.MvNormal('w_team_year_params', mu=np.zeros(2), chol=L_cov_w, shape=(len(team_year_ids), 2))

        u0_driver = u_driver_params[:, 0]
        u1_driver = u_driver_params[:, 1]
        v0_team = v_team_params[:, 0]
        v1_team = v_team_params[:, 1]
        w0_team_year = w_team_year_params[:, 0]
        w1_team_year = w_team_year_params[:, 1]

        mu_random_intercept = u0_driver[driver_idx] + v0_team[team_idx] + w0_team_year[team_year_idx]
        age_effect_component = age_binned_coeffs[age_bin_indices]
        experience_effect_component = experience_coeffs[experience_indices]
        customer_car_effect_component = beta_customer * is_customer_car_flags
        mu_fixed_covariates = beta_1 * num_drivers_in_race + beta_2 * race_competitiveness
        mu_year_interaction = (beta_3 + u1_driver[driver_idx] + v1_team[team_idx] + w1_team_year[team_year_idx]) * year_centered
        mu = (beta_0 + mu_random_intercept + age_effect_component + experience_effect_component + 
              customer_car_effect_component + mu_fixed_covariates + mu_year_interaction)
        points_like = pm.Normal('points_like', mu=mu, sigma=sigma_e, observed=rankit_points)
    return model_year_vf_extended, driver_ids, team_ids, team_year_ids

def get_init_vals_from_trace(trace, model):
    if trace is None or not hasattr(trace, 'posterior'): return None
    init_vals = {}
    posterior = trace.posterior
    model_var_names = [var.name for var in model.free_RVs]
    for var_name in model_var_names:
        if var_name in posterior.data_vars:
            try: init_vals[var_name] = posterior[var_name].isel(chain=0, draw=-1).values
            except Exception as e: print(f"Warning: Could not extract initval for '{var_name}': {e}")
    return init_vals if init_vals else None

def calculate_variance_functions_year(trace_year_vf, years, year_mean_for_centering):
    posterior = trace_year_vf.posterior
    try:
        sigma_u_vec_samples = posterior['u_chol_corr_sigmas_stds'].values
        sigma_v_vec_samples = posterior['v_chol_corr_sigmas_stds'].values
        sigma_w_vec_samples = posterior['w_chol_corr_sigmas_stds'].values
        corr_u_matrix_samples = posterior['u_chol_corr_sigmas_corr'].values
        corr_v_matrix_samples = posterior['v_chol_corr_sigmas_corr'].values
        corr_w_matrix_samples = posterior['w_chol_corr_sigmas_corr'].values
    except KeyError as e:
        print(f"!!! Error extracting LKJ variables for variance: {e}.")
        return None, None, None
    sigma_u0, sigma_u1 = sigma_u_vec_samples[..., 0].flatten(), sigma_u_vec_samples[..., 1].flatten()
    rho_u01 = corr_u_matrix_samples[..., 0, 1].flatten()
    sigma_v0, sigma_v1 = sigma_v_vec_samples[..., 0].flatten(), sigma_v_vec_samples[..., 1].flatten()
    rho_v01 = corr_v_matrix_samples[..., 0, 1].flatten()
    sigma_w0, sigma_w1 = sigma_w_vec_samples[..., 0].flatten(), sigma_w_vec_samples[..., 1].flatten()
    rho_w01 = corr_w_matrix_samples[..., 0, 1].flatten()
    variances_driver, variances_team, variances_team_year = [], [], []
    for year in years:
        year_c = year - year_mean_for_centering
        var_driver = sigma_u0**2 + (year_c**2 * sigma_u1**2) + (2 * year_c * (rho_u01 * sigma_u0 * sigma_u1))
        var_team = sigma_v0**2 + (year_c**2 * sigma_v1**2) + (2 * year_c * (rho_v01 * sigma_v0 * sigma_v1))
        var_team_y = sigma_w0**2 + (year_c**2 * sigma_w1**2) + (2 * year_c * (rho_w01 * sigma_w0 * sigma_w1))
        variances_driver.append(np.mean(var_driver))
        variances_team.append(np.mean(var_team))
        variances_team_year.append(np.mean(var_team_y))
    return variances_driver, variances_team, variances_team_year

def extract_driver_career_effects(trace, driver_ids_map, drivers_info_df):
    """
    Extracts posterior means of u0 (base skill) and u1 (skill trend) for each driver.
    """
    print("\n--- Extracting Driver Career Effects (u0, u1) ---")
    posterior = trace.posterior
    try:
        u_driver_params_mean = posterior['u_driver_params'].mean(dim=['chain', 'draw']).values
        u0_driver_all = u_driver_params_mean[:, 0]
        u1_driver_all = u_driver_params_mean[:, 1]
    except KeyError as e:
        print(f"Error extracting 'u_driver_params': {e}")
        return pd.DataFrame()

    effects_df = pd.DataFrame({
        'driverid': driver_ids_map,
        'u0_skill_intercept': u0_driver_all,
        'u1_skill_trend': u1_driver_all
    })
    effects_df = pd.merge(effects_df, drivers_info_df[['driverid', 'forename', 'surname']], on='driverid', how='left')
    print("Driver career effects extracted.")
    return effects_df

def generate_yearly_driver_rankings(trace, prepared_data_for_model, driver_ids_map, year_mean_centering, drivers_info_df):
    print("\n--- Generating Year-by-Year Driver Rankings ---")
    posterior = trace.posterior
    try:
        u_driver_params_mean = posterior['u_driver_params'].mean(dim=['chain', 'draw']).values
        u0_driver_all = u_driver_params_mean[:, 0]
        u1_driver_all = u_driver_params_mean[:, 1]
        age_coeffs_mean = posterior['age_binned_coeffs'].mean(dim=['chain', 'draw']).values
        exp_coeffs_mean = posterior['experience_coeffs'].mean(dim=['chain', 'draw']).values
    except KeyError as e:
        print(f"Error extracting posterior means for yearly ranking: {e}")
        return pd.DataFrame()

    driver_id_to_idx = {driverid: i for i, driverid in enumerate(driver_ids_map)}
    yearly_data_unique = prepared_data_for_model[['driverid', 'year', 'age_bin_idx', 'experience']].drop_duplicates(subset=['driverid', 'year'])
    yearly_rankings_list = []

    for _, row in yearly_data_unique.iterrows():
        driverid, year, age_bin, experience = row['driverid'], row['year'], int(row['age_bin_idx']), int(row['experience'])
        if driverid not in driver_id_to_idx: continue
        driver_model_idx = driver_id_to_idx[driverid]
        u0_d, u1_d = u0_driver_all[driver_model_idx], u1_driver_all[driver_model_idx]
        year_c = year - year_mean_centering
        age_effect = age_coeffs_mean[age_bin] if 0 <= age_bin < len(age_coeffs_mean) else 0
        exp_effect = exp_coeffs_mean[experience] if 0 <= experience < len(exp_coeffs_mean) else 0
        score = u0_d + (u1_d * year_c) + age_effect + exp_effect
        yearly_rankings_list.append({
            'driverid': driverid, 'year': year, 'yearly_performance_score': score,
            'u0_contrib': u0_d, 'u1_year_contrib': u1_d * year_c, 
            'age_contrib': age_effect, 'exp_contrib': exp_effect
        })

    if not yearly_rankings_list: return pd.DataFrame()
    yearly_rankings_df = pd.DataFrame(yearly_rankings_list)
    yearly_rankings_df = pd.merge(yearly_rankings_df, drivers_info_df[['driverid', 'forename', 'surname']], on='driverid', how='left')
    yearly_rankings_df['yearly_rank'] = yearly_rankings_df.groupby('year')['yearly_performance_score'].rank(method='dense', ascending=False)
    yearly_rankings_df = yearly_rankings_df.sort_values(by=['year', 'yearly_rank'])
    print("--- Finished Generating Year-by-Year Driver Rankings ---")
    return yearly_rankings_df

def save_to_db(df, table_name, engine, if_exists='replace'):
    """Saves a DataFrame to a database table."""
    if df is None or df.empty:
        print(f"DataFrame for table '{table_name}' is empty. Skipping save.")
        return
    try:
        df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
        print(f"Successfully saved DataFrame to table '{table_name}' in results DB.")
    except Exception as e:
        print(f"!!! Error saving DataFrame to table '{table_name}': {e}")

def generate_per_race_rankings(trace, prepared_data, driver_ids_map, team_ids_map, 
                               team_year_ids_map, year_mean_centering, drivers_info_df):
    """
    Generates model-predicted performance rankings for each driver within each race.
    """
    print("\n--- Generating Per-Race Driver Rankings ---")
    posterior = trace.posterior

    # 1. Extract posterior mean for all relevant parameters
    try:
        # Fixed effects
        beta_0_mean = posterior['beta_0'].mean().item()
        beta_1_mean = posterior['beta_1'].mean().item() # num_drivers_in_race
        beta_2_mean = posterior['beta_2'].mean().item() # race_competitiveness
        beta_3_mean = posterior['beta_3'].mean().item() # year_trend
        beta_customer_mean = posterior['beta_customer'].mean().item()

        # Binned effects
        age_coeffs_mean = posterior['age_binned_coeffs'].mean(dim=['chain', 'draw']).values
        exp_coeffs_mean = posterior['experience_coeffs'].mean(dim=['chain', 'draw']).values

        # Random effects (parameters for each driver/team/team-year)
        u_params_mean = posterior['u_driver_params'].mean(dim=['chain', 'draw']).values
        v_params_mean = posterior['v_team_params'].mean(dim=['chain', 'draw']).values
        w_params_mean = posterior['w_team_year_params'].mean(dim=['chain', 'draw']).values
    except KeyError as e:
        print(f"!!! Error extracting posterior means for per-race ranking: {e}")
        return pd.DataFrame()

    # 2. Create efficient ID-to-index mappings for the model
    driver_id_to_idx = {driverid: i for i, driverid in enumerate(driver_ids_map)}
    team_id_to_idx = {teamid: i for i, teamid in enumerate(team_ids_map)}
    team_year_id_to_idx = {ty_id: i for i, ty_id in enumerate(team_year_ids_map)}

    # 3. Iterate through the data to calculate the score for each driver in each race
    # We will work on a copy to avoid modifying the original DataFrame
    ranking_data = prepared_data.copy()
    
    # Define a function to apply to each row
    def calculate_score(row):
        # Get model indices
        driver_idx = driver_id_to_idx.get(row['driverid'])
        team_idx = team_id_to_idx.get(row['teamid'])
        team_year_idx = team_year_id_to_idx.get(row['team_year'])

        # Skip if any entity was not in the model's training data
        if driver_idx is None or team_idx is None or team_year_idx is None:
            return np.nan
        
        # Centered year
        year_c = row['year'] - year_mean_centering

        # --- Reconstruct mu from the model definition ---
        
        # Random Intercepts
        u0_driver = u_params_mean[driver_idx, 0]
        v0_team = v_params_mean[team_idx, 0]
        w0_team_year = w_params_mean[team_year_idx, 0]
        mu_random_intercept = u0_driver + v0_team + w0_team_year
        
        # Binned effects
        age_effect = age_coeffs_mean[int(row['age_bin_idx'])]
        exp_effect = exp_coeffs_mean[int(row['experience'])]

        # Customer car effect
        customer_car_effect = beta_customer_mean * row['is_customer_car_effect_active']
        
        # Fixed covariates (race-specific)
        mu_fixed_covariates = (beta_1_mean * row['num_drivers_in_race'] + 
                               beta_2_mean * row['race_competitiveness'])

        # Year interaction (trends)
        u1_driver = u_params_mean[driver_idx, 1]
        v1_team = v_params_mean[team_idx, 1]
        w1_team_year = w_params_mean[team_year_idx, 1]
        mu_year_interaction = (beta_3_mean + u1_driver + v1_team + w1_team_year) * year_c

        # Combine all components to get the final score
        score = (beta_0_mean + mu_random_intercept + age_effect + exp_effect + 
                 customer_car_effect + mu_fixed_covariates + mu_year_interaction)
        
        return score

    print("Calculating performance score for each driver in each race...")
    ranking_data['race_performance_score'] = ranking_data.apply(calculate_score, axis=1)

    # 4. Rank drivers within each race based on the calculated score
    print("Ranking drivers within each race...")
    ranking_data.dropna(subset=['race_performance_score'], inplace=True)
    ranking_data['predicted_rank'] = ranking_data.groupby('raceid')['race_performance_score'] \
                                                .rank(method='dense', ascending=False)
    
    # 5. Clean up and merge for final output
    final_df = pd.merge(ranking_data, drivers_info_df[['driverid', 'forename', 'surname']], 
                        on='driverid', how='left')
    
    # Select and reorder columns for clarity
    output_cols = [
        'raceid', 'year', 'driverid', 'forename', 'surname', 
        'predicted_rank', 'position', # 'position' is the actual rank
        'race_performance_score', 'teamid'
    ]
    final_df = final_df[output_cols].sort_values(by=['raceid', 'predicted_rank'])

    print("--- Finished Generating Per-Race Driver Rankings ---")
    return final_df

def predict_teammate_head_to_head(trace, driver1_name, driver2_name, team_name, future_year,
                                  prepared_data, drivers_df, driver_ids_map, team_ids_map, team_year_ids_map,
                                  year_mean_centering):
    """
    Predicts the head-to-head performance of two future teammates using the model's posterior.
    """
    print(f"\n--- Predicting Head-to-Head: {driver1_name} vs. {driver2_name} at {team_name} in {future_year} ---")

    # --- 1. Find Entity IDs from Names ---
    def get_driver_id(name, df):
        parts = name.split()
        forename, surname = parts[0], " ".join(parts[1:])
        driver = df[(df['forename'].str.lower() == forename.lower()) & (df['surname'].str.lower() == surname.lower())]
        if driver.empty: return None
        return driver.iloc[0]['driverid']

    driver1_id = get_driver_id(driver1_name, drivers_df)
    driver2_id = get_driver_id(driver2_name, drivers_df)
    
    # Get team ID from the prepared_data based on teamid associated with the team name
    # (This is a bit indirect, would be better to have a constructors_df with name -> constructorid)
    # Assuming team_name matches a name in constructors_df from the beginning
    team_id = constructors_df[constructors_df['name'].str.lower() == team_name.lower()].iloc[0]['constructorid']

    if not all([driver1_id, driver2_id, team_id]):
        print("!!! Error: Could not find one of the drivers or the team in the data.")
        return None

    # --- 2. Get Model Indices and Future Context ---
    driver_id_to_idx = {did: i for i, did in enumerate(driver_ids_map)}
    team_id_to_idx = {tid: i for i, tid in enumerate(team_ids_map)}
    team_year_id_to_idx = {tyid: i for i, tyid in enumerate(team_year_ids_map)}

    d1_idx = driver_id_to_idx.get(driver1_id)
    d2_idx = driver_id_to_idx.get(driver2_id)
    t_idx = team_id_to_idx.get(team_id)
    
    # For team-year, we must make an assumption. The most reasonable one is to use the
    # effect from the team's most recent year in the dataset as the starting point.
    most_recent_team_year = prepared_data[prepared_data['teamid'] == team_id]['year'].max()
    team_year_str = f"{team_id}_{most_recent_team_year}"
    ty_idx = team_year_id_to_idx.get(team_year_str)

    if not all([d1_idx, d2_idx, t_idx, ty_idx]):
        print("!!! Error: Could not find one of the entities in the model's learned parameters.")
        return None

    # --- 3. Calculate Future Age and Experience ---
    def get_future_context(driver_id, year):
        last_entry = prepared_data[prepared_data['driverid'] == driver_id].sort_values('year').iloc[-1]
        last_year, last_age, last_exp = last_entry['year'], last_entry['age'], last_entry['experience']
        
        years_diff = year - last_year
        future_age = last_age + years_diff
        future_exp = min(4, last_exp + years_diff) # Cap experience at 4 as per model structure
        
        # Determine age bin
        age_knots = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47]
        age_bin_edges = [-np.inf] + age_knots + [np.inf]
        age_bin_idx = pd.cut([future_age], bins=age_bin_edges, labels=False, right=False)[0]
        
        return int(age_bin_idx), int(future_exp)

    d1_age_bin, d1_exp = get_future_context(driver1_id, future_year)
    d2_age_bin, d2_exp = get_future_context(driver2_id, future_year)
    print(f"{driver1_name} in {future_year}: Age Bin Idx={d1_age_bin}, Exp Level={d1_exp}")
    print(f"{driver2_name} in {future_year}: Age Bin Idx={d2_age_bin}, Exp Level={d2_exp}")

    # --- 4. Get Posterior Samples and Define Average Race ---
    posterior = trace.posterior
    flat_posterior = {var: posterior[var].stack(sample=("chain", "draw")).values for var in posterior.data_vars}
    
    # Assume an "average" race from the last season
    last_season = prepared_data['year'].max()
    avg_num_drivers = prepared_data[prepared_data['year'] == last_season]['num_drivers_in_race'].mean()
    avg_race_comp = prepared_data[prepared_data['year'] == last_season]['race_competitiveness'].mean()
    year_c = future_year - year_mean_centering

    # --- 5. Construct `mu` for Each Driver Across All Samples ---
    def calculate_mu_dist(d_idx, age_bin, exp):
        mu_rand_int = (flat_posterior['u_driver_params'][:, d_idx, 0] + 
                       flat_posterior['v_team_params'][:, t_idx, 0] + 
                       flat_posterior['w_team_year_params'][:, ty_idx, 0])
                       
        age_effect = flat_posterior['age_binned_coeffs'][:, age_bin]
        exp_effect = flat_posterior['experience_coeffs'][:, exp]
        
        mu_fixed = (flat_posterior['beta_1'] * avg_num_drivers +
                    flat_posterior['beta_2'] * avg_race_comp)
        
        mu_year_int = ((flat_posterior['beta_3'] + 
                        flat_posterior['u_driver_params'][:, d_idx, 1] +
                        flat_posterior['v_team_params'][:, t_idx, 1] +
                        flat_posterior['w_team_year_params'][:, ty_idx, 1]) * year_c)

        # We ignore customer car effect and grid effect for a pure head-to-head skill comparison
        mu_dist = (flat_posterior['beta_0'] + mu_rand_int + age_effect + exp_effect + mu_fixed + mu_year_int)
        return mu_dist

    mu1_dist = calculate_mu_dist(d1_idx, d1_age_bin, d1_exp)
    mu2_dist = calculate_mu_dist(d2_idx, d2_age_bin, d2_exp)

    # --- 6. Compare Distributions and Summarize ---
    delta_dist = mu1_dist - mu2_dist # driver1's performance score minus driver2's

    prob_d1_outperforms = (delta_dist > 0).mean()
    mean_gap = delta_dist.mean()
    hdi_gap = az.hdi(delta_dist, hdi_prob=0.94)

    results = {
        "prediction_for": f"{driver1_name} vs {driver2_name} at {team_name} in {future_year}",
        "prob_d1_outperforms": f"{prob_d1_outperforms:.2%}",
        "prob_d2_outperforms": f"{1 - prob_d1_outperforms:.2%}",
        "avg_performance_gap": f"{mean_gap:.4f} (positive means D1 favored)",
        "gap_94_hdi": [round(x, 4) for x in hdi_gap.tolist()]
    }
    
    return results

# --- Main Execution ---
if __name__ == '__main__':
    results_df, drivers_df, constructors_df = fetch_data_from_sql()
    if results_df is None: raise ValueError("Failed to fetch data.")

    results_df_f1_era = results_df[results_df['year'] >= 1950].copy()
    if results_df_f1_era.empty: raise ValueError("No data found for F1 era (1950 onwards).")
    
    prepared_data = prepare_data(results_df_f1_era, drivers_df, constructors_df)
    if prepared_data is None or prepared_data.empty:
        raise ValueError("Data preparation resulted in an empty DataFrame.")

    print("\n--- Running Extended Year Variance Function Model ---")
    year_mean_for_centering_in_model = prepared_data['year'].mean() 
    print(f"Using year_mean_for_centering = {year_mean_for_centering_in_model} for model and variance calcs.")

    model_year_vf_ext, driver_ids_year_vf, team_ids_year_vf, team_year_ids_year_vf = \
        create_model_year_variance_function(prepared_data.copy())

    trace_initial_year_vf_ext = None
    print("Running Initial MCMC for Extended Year VF Model (short run)...")
    try:
        with model_year_vf_ext:
             trace_initial_year_vf_ext = pm.sample(draws=1000, tune=1000, cores=2, random_seed=123,
                                                   target_accept=0.95, progressbar=True)
    except Exception as e: print(f"Error during initial sampling: {e}")

    init_vals_year_vf_ext = get_init_vals_from_trace(trace_initial_year_vf_ext, model_year_vf_ext)

    trace_year_vf_ext = None
    print("Running Final MCMC for Extended Year VF Model (long run)...")
    FINAL_DRAWS, FINAL_TUNE = 5000, 3000 # Adjust for real runs
    try:
        with model_year_vf_ext:
             trace_year_vf_ext = pm.sample(draws=FINAL_DRAWS, tune=FINAL_TUNE, cores=4, random_seed=789,
                                           initvals=init_vals_year_vf_ext if init_vals_year_vf_ext else 'auto',
                                           target_accept=0.95, progressbar=True)
    except Exception as e: print(f"Error during final sampling: {e}")

    if trace_year_vf_ext:
        print("\n--- Extended Year Variance Model Results ---")
        
        # Save Model Summary
        key_vars_extended = ['beta_0', 'beta_1', 'beta_2', 'beta_3', 'beta_customer',
                             'age_binned_coeffs', 'experience_coeffs',
                             'u_chol_corr_sigmas_stds', 'u_chol_corr_sigmas_corr',
                             'v_chol_corr_sigmas_stds', 'v_chol_corr_sigmas_corr',
                             'w_chol_corr_sigmas_stds', 'w_chol_corr_sigmas_corr', 'sigma_e']
        summary_year_vf_ext_df = az.summary(trace_year_vf_ext, round_to=4, var_names=key_vars_extended)
        print("\nArviZ Summary (Extended Year VF Model - Key Parameters):")
        print(summary_year_vf_ext_df)
        # Reset index to make variable names a column for DB saving
        save_to_db(summary_year_vf_ext_df.reset_index().rename(columns={'index':'variable'}), 
                   'model_summary_extended', db_engine_results)

        # Calculate and Save Variance Functions
        years_for_variance = np.sort(prepared_data['year'].unique())
        variances_driver_ext, variances_team_ext, variances_team_year_ext = \
            calculate_variance_functions_year(trace_year_vf_ext, years_for_variance, year_mean_for_centering_in_model)
        if variances_driver_ext is not None:
            variance_df_year_ext = pd.DataFrame({
                'Year': years_for_variance, 'Driver_Variance': variances_driver_ext,
                'Team_Variance': variances_team_ext, 'Team_Year_Variance': variances_team_year_ext
            })
            print("\nVariance Functions over Years (Extended Model):")
            print(variance_df_year_ext.head())
            save_to_db(variance_df_year_ext, 'variance_functions_yearly', db_engine_results)

        # Extract and Save Driver Career Effects (u0, u1)
        driver_career_effects_df = extract_driver_career_effects(trace_year_vf_ext, driver_ids_year_vf, drivers_df)
        if not driver_career_effects_df.empty:
            print("\nDriver Career Effects (u0, u1) - Sample:")
            print(driver_career_effects_df.head())
            save_to_db(driver_career_effects_df, 'driver_career_effects', db_engine_results)
            
            # For all-time ranking based purely on u0_skill_intercept
            all_time_ranking_u0 = driver_career_effects_df.sort_values(by='u0_skill_intercept', ascending=False)
            print("\nTop 20 All-Time Driver Ranking (based on u0_skill_intercept):")
            print(all_time_ranking_u0[['forename', 'surname', 'u0_skill_intercept']].head(20))
            save_to_db(all_time_ranking_u0, 'driver_all_time_ranking_u0', db_engine_results)


        # Generate and Save Yearly Driver Rankings
        all_yearly_rankings_df = generate_yearly_driver_rankings(
            trace_year_vf_ext, prepared_data.copy(), driver_ids_year_vf, 
            year_mean_for_centering_in_model, drivers_df
        )
        if not all_yearly_rankings_df.empty:
            print("\nSample Yearly Driver Rankings (Top 3 per sample year):")
            sample_years_to_show = sorted(all_yearly_rankings_df['year'].unique())
            if len(sample_years_to_show) > 3:
                sample_years_to_show = [sample_years_to_show[0], sample_years_to_show[len(sample_years_to_show)//2], sample_years_to_show[-1]]
            
            for year_to_show in sample_years_to_show:
                print(f"\n--- Top Drivers in {year_to_show} ---")
                print(all_yearly_rankings_df[all_yearly_rankings_df['year'] == year_to_show].head(3)[
                    ['yearly_rank', 'forename', 'surname', 'yearly_performance_score']
                ])
            save_to_db(all_yearly_rankings_df, 'driver_yearly_performance_rankings', db_engine_results)
        
        # Save Age and Experience Effects Coefficients
        posterior = trace_year_vf_ext.posterior
        try:
            age_coeffs_mean = posterior['age_binned_coeffs'].mean(dim=['chain', 'draw']).values
            age_knots_summary = [-np.inf] + [20, 23, 26, 29, 32, 35, 38, 41, 44, 47] + [np.inf]
            age_bin_labels = []
            for i in range(len(age_knots_summary) - 1):
                lower, upper = age_knots_summary[i], age_knots_summary[i+1]
                lower_str = "<" if lower == -np.inf else f"[{int(lower)},"
                upper_str = "inf)" if upper == np.inf else f"{int(upper)})"
                age_bin_labels.append(f"Age {lower_str}{upper_str}")
            age_effects_df = pd.DataFrame({'age_bin_label': age_bin_labels, 'coefficient': age_coeffs_mean})
            print("\nAge Effects Coefficients:")
            print(age_effects_df)
            save_to_db(age_effects_df, 'age_effects_coefficients', db_engine_results)

            exp_coeffs_mean = posterior['experience_coeffs'].mean(dim=['chain', 'draw']).values
            exp_labels = [f"Exp_{i}yr" for i in range(len(exp_coeffs_mean))] # Should be 5 levels (0-4)
            exp_effects_df = pd.DataFrame({'experience_level': exp_labels, 'coefficient': exp_coeffs_mean})
            print("\nExperience Effects Coefficients:")
            print(exp_effects_df)
            save_to_db(exp_effects_df, 'experience_effects_coefficients', db_engine_results)
        except KeyError as e:
            print(f"Could not extract age/experience coefficients for saving: {e}")

        per_race_rankings_df = generate_per_race_rankings(
            trace_year_vf_ext,
            prepared_data.copy(),
            driver_ids_year_vf,
            team_ids_year_vf,
            team_year_ids_year_vf,
            year_mean_for_centering_in_model,
            drivers_df
        )


        if not per_race_rankings_df.empty:
            
            # Find a recent race to display for a more interesting example
            recent_race_id = per_race_rankings_df['raceid'].max()
            print(f"\nModel's Predicted Ranking for the most recent race (ID: {recent_race_id}):")
            print(per_race_rankings_df[per_race_rankings_df['raceid'] == recent_race_id])

            save_to_db(per_race_rankings_df, 'driver_per_race_rankings', db_engine_results)

        # --- PREDICTION SECTION ---
        prediction_results = predict_teammate_head_to_head(
            trace=trace_year_vf_ext,
            driver1_name="Charles Leclerc",
            driver2_name="Lewis Hamilton",
            team_name="Ferrari",
            future_year=2025,
            prepared_data=prepared_data,
            drivers_df=drivers_df,
            driver_ids_map=driver_ids_year_vf,
            team_ids_map=team_ids_year_vf,
            team_year_ids_map=team_year_ids_year_vf,
            year_mean_centering=year_mean_for_centering_in_model
        )

        if prediction_results:
            print("\n--- Future Prediction Results ---")
            for key, value in prediction_results.items():
                print(f"{key}: {value}")



    else:
        print("Skipping Extended Year VF Model analysis because trace_year_vf_ext is missing.")

    print("\nScript execution finished.")
