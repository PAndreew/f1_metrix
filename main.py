import pymc as pm
import pandas as pd
import numpy as np
import unicodedata
import arviz as az
import os
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
    Fetches and combines data from all necessary tables in a single, robust SQL query.
    """
    print("--- Fetching Data from SQL ---")

    # CORRECTED QUERY: Uses the specific 'type' values from your database schema,
    # including the crucial 'STARTING_GRID_POSITION'.
    combined_query = """
    WITH GridQualiData AS (
        SELECT
            race_id,
            driver_id,
            -- Use MIN to handle cases where a driver might have multiple entries (e.g., sprint)
            MIN(starting_grid_position_qualification_position_number) as grid_position
        FROM race_data
        WHERE "type" IN (
            'STARTING_GRID_POSITION', -- The type you correctly identified
            'STARTING_GRID',          -- Including for robustness, in case it's used elsewhere
            'QUALIFYING_RESULT',      -- Often contains grid data
            'SPRINT_GRID'             -- For modern F1 sprint race weekends
        )
          -- Ensure we only pull rows that actually have a grid position
          AND starting_grid_position_qualification_position_number IS NOT NULL
        GROUP BY race_id, driver_id
    )
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
        r.circuit_id AS circuitid,
        c.type AS circuit_type,
        rd.position_text AS position_description,
        rd.race_reason_retired AS dnf_reason,
        gq.grid_position AS grid, -- Get the grid position from our CTE
        -- This subquery to get qualifying time is fine to keep
        (SELECT MIN(qt.qualifying_time) FROM race_data qt WHERE qt.race_id = rd.race_id AND qt.driver_id = rd.driver_id AND qt."type" = 'QUALIFYING_RESULT') as qualifying_time
    FROM race_data rd
    JOIN race r ON rd.race_id = r.id
    JOIN circuit c ON r.circuit_id = c.id
    -- Use a LEFT JOIN so we still get race results even if grid data is missing for some older races
    LEFT JOIN GridQualiData gq ON rd.race_id = gq.race_id AND rd.driver_id = gq.driver_id
    WHERE rd."type" = 'RACE_RESULT'
      AND r.grand_prix_id NOT IN (SELECT id FROM grand_prix WHERE full_name LIKE '%Indianapolis 500%');
    """
    try:
        results_df = pd.read_sql_query(combined_query, db_engine)
        print(f"Fetched {len(results_df)} rows from race_data (type='RACE_RESULT') with grid data joined.")
    except Exception as e:
        print(f"!!! Error executing the combined SQL query: {e}")
        return None, None, None

    # This check is now the most important one. It should show a high number.
    print(f"Grid column has {results_df['grid'].notna().sum()} non-null values after improved query.")
    if results_df['grid'].notna().sum() == 0:
        print("!!! CRITICAL WARNING: The grid column is still empty. Check the 'type' values in the SQL query again!")

    # --- The rest of your function (processing dtypes, fetching drivers, etc.) remains the same ---
    
    if 'position' in results_df.columns:
        results_df.dropna(subset=['position'], inplace=True)
        results_df['position'] = pd.to_numeric(results_df['position'], errors='coerce').astype('Int64')
        if results_df['position'].isnull().any():
            print("!!! Warning: NULL values in 'position' after numeric conversion. Dropping these rows.")
            results_df.dropna(subset=['position'], inplace=True)
            results_df['position'] = results_df['position'].astype(int) 
    else:
        print("!!! Critical Warning: 'position' column missing.")
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
    data = race_results_df.copy()

    # --- (Your existing age and experience calculations are good) ---
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

    data['points'] = data['position'].apply(calculate_points_robust)
    data = data.groupby(['driverid', 'year'], group_keys=False).apply(robustify_points_season)
    data['rankit_points'] = data.groupby('raceid')['points'].transform(lambda x: rankit(x.values))
    
    # --- (Your feature creation logic) ---
    data['num_drivers_in_race'] = data.groupby('raceid')['driverid'].transform('count')
    data['norm_finish_pos'] = np.where(data['num_drivers_in_race'] > 0, data['position'] / data['num_drivers_in_race'], np.nan)
    driver_avg_norm_pos = data.groupby('driverid')['norm_finish_pos'].mean().rename('driver_avg_norm_pos')
    data = pd.merge(data, driver_avg_norm_pos, on='driverid', how='left')
    race_competitiveness = data.groupby('raceid')['driver_avg_norm_pos'].mean().rename('race_competitiveness')
    data = pd.merge(data, race_competitiveness, on='raceid', how='left')
    data['team_year'] = data['teamid'].astype(str) + '_' + data['year'].astype(str)

    
    print("Preparing grid position variable...")
    # --- START REPLACEMENT BLOCK ---
    
    # Calculate the median grid position for each race. This will be a float.
    median_grid_per_race = data.groupby('raceid')['grid'].transform('median')

    # >> FIX: Before filling, round the median to the nearest integer and cast it
    # to the same nullable integer type as the target column.
    # This prevents the float-to-int casting error.
    median_grid_per_race_int = median_grid_per_race.round().astype('Int64')

    # Create the new column for the model
    data['grid_for_model'] = data['grid'].copy()
    
    # Now, fill the NaNs in the new column using the correctly typed integer medians.
    data['grid_for_model'].fillna(median_grid_per_race_int, inplace=True)
    
    # For any races where ALL grid positions were missing (so median is also NaN),
    # fall back to the last-place method.
    data['grid_for_model'].fillna(data['num_drivers_in_race'] + 1, inplace=True)
    
    # Handle grid positions of 0 (e.g., pit lane start) by setting them to a high number.
    data.loc[data['grid_for_model'] == 0, 'grid_for_model'] = data['num_drivers_in_race'] + 1
    
    # Ensure grid is always at least 1 before taking the log to prevent -inf.
    # We must explicitly convert to a numeric type that isn't nullable for the comparison.
    data['grid_for_model'] = pd.to_numeric(data['grid_for_model'], errors='coerce')
    data['grid_for_model'] = data['grid_for_model'].apply(lambda x: max(x, 1))

    # Check for any remaining issues before the log transform
    if data['grid_for_model'].isnull().any() or (data['grid_for_model'] <= 0).any():
        print("!!! FATAL: Invalid values still found in 'grid_for_model' before log transform.")
        data['grid_for_model'].fillna(data['num_drivers_in_race'] + 1, inplace=True)
        assert (data['grid_for_model'] > 0).all(), "Grid for model must be positive!"

    data['log_grid'] = np.log(data['grid_for_model'])

    age_knots = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47]
    age_bin_edges = [-np.inf] + age_knots + [np.inf]
    data['age_numeric'] = pd.to_numeric(data['age'], errors='coerce')
    data['age_bin_idx'] = pd.cut(data['age_numeric'], bins=age_bin_edges, labels=False, right=False)
    if data['age_bin_idx'].isnull().any():
        print(f"Warning: {data['age_bin_idx'].isnull().sum()} entries have NaN 'age_bin_idx'.")

    # The number of age bins is 11 (0-10)
    assert data['age_bin_idx'].max() <= 10, "Age bin index is out of bounds!"
    assert data['age_bin_idx'].min() >= 0, "Age bin index is out of bounds!"
    # The number of experience levels is 5 (0-4)
    assert data['experience'].max() <= 4, "Experience index is out of bounds!"
    assert data['experience'].min() >= 0, "Experience index is out of bounds!"

    print("Standardizing continuous predictors...")
    
    cols_to_standardize = ['num_drivers_in_race', 'race_competitiveness', 'log_grid', 'year']
    standardization_params = {}
    
    for col in cols_to_standardize:
        mean = data[col].mean()
        std = data[col].std()
        if std == 0:
            std = 1 
        standardization_params[col] = {'mean': mean, 'std': std}
        
        # Standardize
        data[f'{col}_std'] = (data[col] - mean) / std
        
        # >> FIX: Clip the standardized values to a robust range [-4, 4]
        # This prevents extreme outliers from causing numerical instability in the model.
        data[f'{col}_std'] = data[f'{col}_std'].clip(lower=-4, upper=4)

    essential_cols = ['rankit_points', 'driverid', 'teamid', 'team_year',
                      'year', 'age_bin_idx', 'experience',
                      'num_drivers_in_race_std', 'race_competitiveness_std', 'log_grid_std', 'year_std']

    initial_rows = len(data)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=essential_cols, inplace=True)
    
    data['age_bin_idx'] = data['age_bin_idx'].astype(int)
    data['experience'] = data['experience'].astype(int)

    print(f"Dropped {initial_rows - len(data)} rows due to NaNs/Infs in essential columns.")
    if data.empty: raise ValueError("DataFrame is empty after preparation.")
    print("--- Finished Preparing Data ---")

    print("--- Checking standardized column distributions ---")
    print(data[['num_drivers_in_race_std', 'race_competitiveness_std', 'log_grid_std', 'year_std']].describe())

    return data, standardization_params

def debug_model_initialization(model):
    """
    Manually evaluates the model's log probability at the initial point
    to find the source of any NaN/inf values.
    """
    print("\n--- RUNNING MODEL INITIALIZATION DEBUGGER ---")
    
    # 1. Get the function that calculates the log probability
    logp_func = model.compile_logp()

    # 2. Get the initial values the sampler would start with
    initial_point = model.initial_point()

    # 3. Evaluate the logp at that starting point
    print("Evaluating log-probability at the initial point...")
    try:
        initial_logp = logp_func(initial_point)
        print(f"Initial Log-Probability: {initial_logp}")
        if np.isnan(initial_logp) or np.isinf(initial_logp):
            print("!!! FATAL: Log-probability is NaN or Inf. Investigating components...")
        else:
            print("--- Initial log-probability is a valid number. The issue might be in the gradient calculation. ---")

    except Exception as e:
        print(f"!!! FATAL: An error occurred during logp evaluation: {e}")
        print("This often means a value was so large it couldn't even be processed.")

    # 4. If the logp was bad, we inspect every variable's contribution
    print("\n--- Inspecting Individual Variable Contributions ---")
    # This loop iterates through every single variable in the model
    for var in model.value_vars:
        try:
            var_logp_func = model.compile_logp(vars=[var])
            var_logp_val = var_logp_func(initial_point)
            
            # Check if this specific variable's contribution is the problem
            if np.isnan(var_logp_val) or np.isinf(var_logp_val):
                print(f"!!! CULPRIT FOUND: The logp of variable '{var.name}' is {var_logp_val}")
                
                # Let's see the initial value that caused the problem
                initial_var_value = initial_point.get(var.name)
                print(f"    > Problem occurred with initial value(s) for '{var.name}':")
                print(f"      Shape: {initial_var_value.shape}, Min: {np.min(initial_var_value):.4f}, Max: {np.max(initial_var_value):.4f}, Mean: {np.mean(initial_var_value):.4f}")
                
            else:
                # This one is fine
                pass # print(f"Variable '{var.name}' is OK (logp: {var_logp_val:.2f})")

        except Exception as e:
            print(f"!!! CULPRIT FOUND (via exception): Error evaluating logp for variable '{var.name}': {e}")
            initial_var_value = initial_point.get(var.name)
            print(f"    > Problem occurred with initial value(s) for '{var.name}':")
            print(f"      Shape: {initial_var_value.shape}, Min: {np.min(initial_var_value):.4f}, Max: {np.max(initial_var_value):.4f}, Mean: {np.mean(initial_var_value):.4f}")

    print("--- FINISHED MODEL INITIALIZATION DEBUGGER ---\n")

def create_model_year_variance_function(data):
    # This part of the function, where you get the raw arrays, stays the same.
    driver_ids = data['driverid'].unique()
    team_ids = data['teamid'].unique()
    team_year_ids = data['team_year'].unique()
    
    expected_age_bins = 11 
    num_experience_levels = 5 

    driver_idx_data = pd.Categorical(data['driverid'], categories=driver_ids).codes
    team_idx_data = pd.Categorical(data['teamid'], categories=team_ids).codes
    team_year_idx_data = pd.Categorical(data['team_year'], categories=team_year_ids).codes
    
    age_bin_indices_data = data['age_bin_idx'].to_numpy(dtype='int64') # Use int64 for indices
    experience_indices_data = data['experience'].to_numpy(dtype='int64') # Use int64 for indices

    num_drivers_in_race_std_data = data['num_drivers_in_race_std'].to_numpy(dtype='float64')
    race_competitiveness_std_data = data['race_competitiveness_std'].to_numpy(dtype='float64')
    log_grid_std_data = data['log_grid_std'].to_numpy(dtype='float64')
    year_std_data = data['year_std'].to_numpy(dtype='float64')
    
    rankit_points_data = data['rankit_points'].to_numpy(dtype='float64')

    # ... Your debugging checks for NaN/inf can remain here if you wish ...

    # --- START OF MAJOR CHANGE ---
    with pm.Model() as model_year_vf_extended:
        # >> FIX: Define data as pm.Data containers <<
        # This makes the model more robust and is the standard way for prediction.
        driver_idx = pm.Data('driver_idx', driver_idx_data, mutable=True)
        team_idx = pm.Data('team_idx', team_idx_data, mutable=True)
        team_year_idx = pm.Data('team_year_idx', team_year_idx_data, mutable=True)
        age_bin_indices = pm.Data('age_bin_indices', age_bin_indices_data, mutable=True)
        experience_indices = pm.Data('experience_indices', experience_indices_data, mutable=True)
        
        num_drivers_in_race_std = pm.Data('num_drivers_in_race_std', num_drivers_in_race_std_data, mutable=True)
        race_competitiveness_std = pm.Data('race_competitiveness_std', race_competitiveness_std_data, mutable=True)
        log_grid_std = pm.Data('log_grid_std', log_grid_std_data, mutable=True)
        year_std = pm.Data('year_std', year_std_data, mutable=True)

        # --- Priors for Fixed Effects ---
        beta_0 = pm.Normal('beta_0', mu=0, sigma=1)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=0.5)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=0.5)
        beta_3 = pm.Normal('beta_3', mu=0, sigma=0.5)
        beta_grid = pm.Normal('beta_grid', mu=0, sigma=0.5)
        
        age_binned_coeffs = pm.Normal('age_binned_coeffs', mu=0, sigma=0.5, shape=expected_age_bins)
        experience_coeffs = pm.Normal('experience_coeffs', mu=0, sigma=0.5, shape=num_experience_levels)
        beta_customer = pm.Normal('beta_customer', mu=0, sigma=0.5)
        sigma_e = pm.HalfNormal('sigma_e', sigma=1)

        # --- Priors for Hierarchical Effects (using the stable Exponential prior) ---
        sd_dist = pm.Exponential.dist(1.0)
        
        chol_corr_u, _, sigmas_u = pm.LKJCholeskyCov('u_chol_corr_sigmas', n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)
        L_cov_u = pm.math.dot(pt.diag(sigmas_u), chol_corr_u)
        u_driver_params = pm.MvNormal('u_driver_params', mu=np.zeros(2), chol=L_cov_u, shape=(len(driver_ids), 2))

        chol_corr_v, _, sigmas_v = pm.LKJCholeskyCov('v_chol_corr_sigmas', n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)
        L_cov_v = pm.math.dot(pt.diag(sigmas_v), chol_corr_v)
        v_team_params = pm.MvNormal('v_team_params', mu=np.zeros(2), chol=L_cov_v, shape=(len(team_ids), 2))

        chol_corr_w, _, sigmas_w = pm.LKJCholeskyCov('w_chol_corr_sigmas', n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)
        L_cov_w = pm.math.dot(pt.diag(sigmas_w), chol_corr_w)
        w_team_year_params = pm.MvNormal('w_team_year_params', mu=np.zeros(2), chol=L_cov_w, shape=(len(team_year_ids), 2))

        # --- Model Linear Equation ---
        # The equation part is exactly the same as before.
        u0_driver = u_driver_params[:, 0]
        u1_driver = u_driver_params[:, 1]
        v0_team = v_team_params[:, 0]
        v1_team = v_team_params[:, 1]
        w0_team_year = w_team_year_params[:, 0]
        w1_team_year = w_team_year_params[:, 1]

        mu_random_intercept = u0_driver[driver_idx] + v0_team[team_idx] + w0_team_year[team_year_idx]
        age_effect_component = age_binned_coeffs[age_bin_indices]
        experience_effect_component = experience_coeffs[experience_indices]
        customer_car_effect_component = beta_customer
        
        mu_fixed_covariates = (beta_1 * num_drivers_in_race_std + 
                               beta_2 * race_competitiveness_std + 
                               beta_grid * log_grid_std)
        
        mu_year_interaction = (beta_3 + u1_driver[driver_idx] + v1_team[team_idx] + w1_team_year[team_year_idx]) * year_std
        
        mu = (beta_0 + mu_random_intercept + age_effect_component + experience_effect_component + 
              customer_car_effect_component + mu_fixed_covariates + mu_year_interaction)
        
        # --- Likelihood ---
        # NOTE: The observed data is now passed here
        points_like = pm.Normal('points_like', mu=mu, sigma=sigma_e, observed=rankit_points_data)
        
    return model_year_vf_extended, driver_ids, team_ids, team_year_ids

# **CHANGE**: The function now accepts `standardization_params`
def calculate_performance_over_expectation(trace, prepared_data, driver_ids_map, team_ids_map, team_year_ids_map, drivers_info_df, standardization_params):
    print("\n--- Calculating Performance Over Expectation (Grid-Adjusted) ---")
    posterior = trace.posterior
    try:
        post_means = {var: posterior[var].mean().item() for var in ['beta_0', 'beta_1', 'beta_2', 'beta_3', 'beta_customer', 'beta_grid']}
        age_coeffs_mean = posterior['age_binned_coeffs'].mean(dim=['chain', 'draw']).values
        exp_coeffs_mean = posterior['experience_coeffs'].mean(dim=['chain', 'draw']).values
        u_params_mean = posterior['u_driver_params'].mean(dim=['chain', 'draw']).values
        v_params_mean = posterior['v_team_params'].mean(dim=['chain', 'draw']).values
        w_params_mean = posterior['w_team_year_params'].mean(dim=['chain', 'draw']).values
    except KeyError as e:
        print(f"!!! Error extracting posterior means (check for 'beta_grid'): {e}")
        return pd.DataFrame()

    driver_id_to_idx = {driverid: i for i, driverid in enumerate(driver_ids_map)}
    team_id_to_idx = {teamid: i for i, teamid in enumerate(team_ids_map)}
    team_year_id_to_idx = {ty_id: i for i, ty_id in enumerate(team_year_ids_map)}
    data_with_residuals = prepared_data.copy()

    def calculate_predicted_mu(row):
        driver_idx = driver_id_to_idx.get(row['driverid'])
        team_idx = team_id_to_idx.get(row['teamid'])
        team_year_idx = team_year_id_to_idx.get(row['team_year'])
        if any(idx is None for idx in [driver_idx, team_idx, team_year_idx]): return np.nan
        
        mu_random_intercept = u_params_mean[driver_idx, 0] + v_params_mean[team_idx, 0] + w_params_mean[team_year_idx, 0]
        age_effect = age_coeffs_mean[int(row['age_bin_idx'])]
        exp_effect = exp_coeffs_mean[int(row['experience'])]
        customer_car_effect = post_means['beta_customer']
        
        mu_fixed_covariates = (post_means['beta_1'] * row['num_drivers_in_race_std'] + 
                               post_means['beta_2'] * row['race_competitiveness_std'] + 
                               post_means['beta_grid'] * row['log_grid_std'])
        
        # **CHANGE**: Use the 'year_std' column from the dataframe for consistency.
        year_std_val = row['year_std']
        mu_year_interaction = (post_means['beta_3'] + u_params_mean[driver_idx, 1] + v_params_mean[team_idx, 1] + w_params_mean[team_year_idx, 1]) * year_std_val
        
        return (post_means['beta_0'] + mu_random_intercept + age_effect + exp_effect + customer_car_effect + mu_fixed_covariates + mu_year_interaction)

    print("Calculating grid-adjusted predicted performance (mu)...")
    data_with_residuals['predicted_mu_grid_adjusted'] = data_with_residuals.apply(calculate_predicted_mu, axis=1)
    data_with_residuals['performance_over_expectation'] = data_with_residuals['rankit_points'] - data_with_residuals['predicted_mu_grid_adjusted']
    data_with_residuals.dropna(subset=['performance_over_expectation'], inplace=True)
    
    final_df = pd.merge(data_with_residuals, drivers_info_df[['driverid', 'forename', 'surname']], on='driverid', how='left')
    
    # RENAME the column for a cleaner output name
    # final_df.rename(columns={'grid_final': 'starting_grid'}, inplace=True)
    
    # UPDATE this list to use the new column name
    output_cols = ['raceid', 'year', 'driverid', 'forename', 'surname', 'position', 'grid', 'rankit_points', 'predicted_mu_grid_adjusted', 'performance_over_expectation']
    
    final_df = final_df[output_cols].sort_values(by='performance_over_expectation', ascending=False)
    
    print("--- Finished Calculating Performance Over Expectation ---")
    return final_df

# **CHANGE**: The function now accepts `standardization_params`
def generate_yearly_driver_rankings(trace, prepared_data_for_model, driver_ids_map, standardization_params, drivers_info_df):
    print("\n--- Generating Year-by-Year 'Pure Skill' Driver Rankings ---")
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

    # **CHANGE**: Get year mean and std from standardization_params for correct scaling.
    year_mean = standardization_params['year']['mean']
    year_std = standardization_params['year']['std']
    
    driver_id_to_idx = {driverid: i for i, driverid in enumerate(driver_ids_map)}
    yearly_data_unique = prepared_data_for_model[['driverid', 'year', 'age_bin_idx', 'experience']].drop_duplicates(subset=['driverid', 'year'])
    yearly_rankings_list = []
    
    for _, row in yearly_data_unique.iterrows():
        driverid, year, age_bin, exp = row['driverid'], row['year'], int(row['age_bin_idx']), int(row['experience'])
        if driverid not in driver_id_to_idx: continue
        
        driver_model_idx = driver_id_to_idx[driverid]
        u0_d, u1_d = u0_driver_all[driver_model_idx], u1_driver_all[driver_model_idx]
        
        # **CHANGE**: Calculate the standardized year value to match the model's input.
        year_std_val = (year - year_mean) / year_std
        
        age_effect = age_coeffs_mean[age_bin] if 0 <= age_bin < len(age_coeffs_mean) else 0
        exp_effect = exp_coeffs_mean[exp] if 0 <= exp < len(exp_coeffs_mean) else 0
        
        # **CHANGE**: Use the standardized year in the score calculation.
        score = u0_d + (u1_d * year_std_val) + age_effect + exp_effect
        yearly_rankings_list.append({'driverid': driverid, 'year': year, 'yearly_pure_skill_score': score})
        
    if not yearly_rankings_list: return pd.DataFrame()
    
    yearly_rankings_df = pd.DataFrame(yearly_rankings_list)
    yearly_rankings_df = pd.merge(yearly_rankings_df, drivers_info_df[['driverid', 'forename', 'surname']], on='driverid', how='left')
    yearly_rankings_df['yearly_rank'] = yearly_rankings_df.groupby('year')['yearly_pure_skill_score'].rank(method='dense', ascending=False)
    yearly_rankings_df = yearly_rankings_df.sort_values(by=['year', 'yearly_rank'])
    
    print("--- Finished Generating Year-by-Year 'Pure Skill' Rankings ---")
    return yearly_rankings_df

# **CHANGE**: The function now accepts `standardization_params`
def predict_teammate_head_to_head(trace, driver1_name, driver2_name, team_id_slug, future_year,
                                  prepared_data, drivers_df, constructors_df, driver_ids_map, team_ids_map, 
                                  team_year_ids_map, standardization_params):
    """
    Predicts the head-to-head performance of two future teammates using the model's posterior.
    (Robust version that correctly handles posterior distributions)
    """
    team_record = constructors_df[constructors_df['constructorid'] == team_id_slug]
    team_name_for_print = team_record['name'].iloc[0] if not team_record.empty else team_id_slug

    print(f"\n--- Predicting Head-to-Head: {driver1_name} vs. {driver2_name} at {team_name_for_print} in {future_year} ---")
    
    # --- START OF FIX ---
    def normalize_str(s):
        """Normalizes a string by removing accents and converting to lowercase."""
        if not isinstance(s, str):
            return ""
        return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        ).lower()
    
    def get_driver_id(name, df):
        """Robustly finds a driver ID by normalizing names for comparison."""
        parts = name.split()
        if len(parts) < 2: return None
        
        # Normalize the search terms
        search_forename = normalize_str(parts[0])
        search_surname = normalize_str(" ".join(parts[1:]))
        
        # Create normalized columns in the dataframe to search against
        df['norm_forename'] = df['forename'].apply(normalize_str)
        df['norm_surname'] = df['surname'].apply(normalize_str)
        
        # Perform the robust search
        driver = df[(df['norm_forename'] == search_forename) & (df['norm_surname'] == search_surname)]
        
        # Clean up the temporary columns
        df.drop(columns=['norm_forename', 'norm_surname'], inplace=True)
        
        return driver.iloc[0]['driverid'] if not driver.empty else None

    driver1_id = get_driver_id(driver1_name, drivers_df)
    driver2_id = get_driver_id(driver2_name, drivers_df)
    team_id = team_id_slug if team_id_slug in constructors_df['constructorid'].values else None

    if not all([driver1_id, driver2_id, team_id]):
        print(f"!!! Error: Could not find one of the entities. Found: Driver1({bool(driver1_id)}), Driver2({bool(driver2_id)}), Team({bool(team_id)})")
        return None

    driver_id_to_idx = {did: i for i, did in enumerate(driver_ids_map)}
    team_id_to_idx = {tid: i for i, tid in enumerate(team_ids_map)}
    team_year_id_to_idx = {tyid: i for i, tyid in enumerate(team_year_ids_map)}
    d1_idx, d2_idx, t_idx = driver_id_to_idx.get(driver1_id), driver_id_to_idx.get(driver2_id), team_id_to_idx.get(team_id)
    
    most_recent_team_year = prepared_data[prepared_data['teamid'] == team_id]['year'].max()
    if pd.isna(most_recent_team_year):
        print(f"!!! Error: Team '{team_name_for_print}' has no history in the prepared_data.")
        return None
    team_year_str = f"{team_id}_{int(most_recent_team_year)}"
    ty_idx = team_year_id_to_idx.get(team_year_str)

    if not all([d1_idx is not None, d2_idx is not None, t_idx is not None, ty_idx is not None]):
        print(f"!!! Error: Could not find one of the entities in the model's learned parameters. Found: D1({bool(d1_idx)}), D2({bool(d2_idx)}), T({bool(t_idx)}), TY({bool(ty_idx)})")
        return None

    def get_future_context(driver_id, year):
        last_entry = prepared_data[prepared_data['driverid'] == driver_id].sort_values('year').iloc[-1]
        years_diff = year - last_entry['year']
        future_age = (last_entry['age'] + years_diff) if pd.notna(last_entry['age']) else 30 
        future_exp = min(4, last_entry['experience'] + years_diff)
        age_knots = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47]
        age_bin_idx = pd.cut([future_age], bins=[-np.inf] + age_knots + [np.inf], labels=False, right=False)[0]
        return int(age_bin_idx), int(future_exp)

    d1_age_bin, d1_exp = get_future_context(driver1_id, future_year)
    d2_age_bin, d2_exp = get_future_context(driver2_id, future_year)
    print(f"{driver1_name} in {future_year}: Age Bin Idx={d1_age_bin}, Exp Level={d1_exp}")
    print(f"{driver2_name} in {future_year}: Age Bin Idx={d2_age_bin}, Exp Level={d2_exp}")

    posterior = trace.posterior
    
    # --- Standardize future context variables ---
    last_season = prepared_data['year'].max()
    avg_num_drivers = prepared_data[prepared_data['year'] == last_season]['num_drivers_in_race'].mean()
    avg_race_comp = prepared_data[prepared_data['year'] == last_season]['race_competitiveness'].mean()
    avg_log_grid = prepared_data[prepared_data['year'] == last_season]['log_grid'].mean()

    num_drivers_std = (avg_num_drivers - standardization_params['num_drivers_in_race']['mean']) / standardization_params['num_drivers_in_race']['std']
    race_comp_std = (avg_race_comp - standardization_params['race_competitiveness']['mean']) / standardization_params['race_competitiveness']['std']
    log_grid_std = (avg_log_grid - standardization_params['log_grid']['mean']) / standardization_params['log_grid']['std']
    
    year_mean = standardization_params['year']['mean']
    year_std = standardization_params['year']['std']
    year_std_val = (future_year - year_mean) / year_std

    # This is a helper function to reliably get flattened posterior samples
    def get_flat_posterior(var_name, trace_obj):
        # This ensures the dimensions are always (sample, dim_1, dim_2, ...)
        return trace_obj.posterior[var_name].stack(sample=("chain", "draw")).transpose("sample", ...)

    def calculate_mu_dist(d_idx, age_bin, exp):
        # --- Get full posterior distributions for ALL parameters ---
        beta_0_dist = get_flat_posterior('beta_0', trace).values
        beta_1_dist = get_flat_posterior('beta_1', trace).values
        beta_2_dist = get_flat_posterior('beta_2', trace).values
        beta_3_dist = get_flat_posterior('beta_3', trace).values
        beta_grid_dist = get_flat_posterior('beta_grid', trace).values

        # --- FIX IS HERE: Explicitly get the values and slice ---
        # Get the full posterior array for all drivers
        u_all_drivers_dist = get_flat_posterior('u_driver_params', trace).values
        v_all_teams_dist = get_flat_posterior('v_team_params', trace).values
        w_all_team_years_dist = get_flat_posterior('w_team_year_params', trace).values
        age_all_bins_dist = get_flat_posterior('age_binned_coeffs', trace).values
        exp_all_levels_dist = get_flat_posterior('experience_coeffs', trace).values
        
        # Now, slice the correct dimension (axis=1) for the specific driver/team/etc.
        u_driver_params_dist = u_all_drivers_dist[:, d_idx, :]
        v_team_params_dist = v_all_teams_dist[:, t_idx, :]
        w_team_year_params_dist = w_all_team_years_dist[:, ty_idx, :]
        age_effect_dist = age_all_bins_dist[:, age_bin]
        exp_effect_dist = exp_all_levels_dist[:, exp]

        # --- Assemble the prediction ---
        mu_rand_int = u_driver_params_dist[:, 0] + v_team_params_dist[:, 0] + w_team_year_params_dist[:, 0]
        
        mu_fixed = (beta_1_dist * num_drivers_std + 
                    beta_2_dist * race_comp_std + 
                    beta_grid_dist * log_grid_std)
        
        mu_year_int = (beta_3_dist + u_driver_params_dist[:, 1] + v_team_params_dist[:, 1] + w_team_year_params_dist[:, 1]) * year_std_val
        
        return (beta_0_dist + mu_rand_int + age_effect_dist + exp_effect_dist + mu_fixed + mu_year_int)
    
    # --- Calculate and compare the two driver distributions ---
    mu1_dist = calculate_mu_dist(d1_idx, d1_age_bin, d1_exp)
    mu2_dist = calculate_mu_dist(d2_idx, d2_age_bin, d2_exp)
    
    delta_dist = mu1_dist - mu2_dist
    prob_d1_outperforms = (delta_dist > 0).mean()
    mean_gap, hdi_gap = delta_dist.mean(), az.hdi(delta_dist, hdi_prob=0.94)

    return {
        "prediction_for": f"{driver1_name} vs {driver2_name} at {team_name_for_print} in {future_year}",
        "prob_d1_outperforms": f"{prob_d1_outperforms:.2%}",
        "prob_d2_outperforms": f"{1 - prob_d1_outperforms:.2%}",
        "avg_performance_gap": f"{mean_gap:.4f} (positive means D1 favored)",
        "gap_94_hdi": [round(x, 4) for x in hdi_gap.tolist()]
    }

def save_to_db(df, table_name, engine, if_exists='replace'):
    if df is None or df.empty:
        print(f"DataFrame for table '{table_name}' is empty. Skipping save.")
        return
    try:
        df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
        print(f"Successfully saved DataFrame to table '{table_name}' in results DB.")
    except Exception as e:
        print(f"!!! Error saving DataFrame to table '{table_name}': {e}")

# --- Main Execution ---
if __name__ == '__main__':
    results_df, drivers_df, constructors_df = fetch_data_from_sql()
    if results_df is None: raise ValueError("Failed to fetch data.")
    
    results_df_f1_era = results_df[results_df['year'] >= 1950].copy()
    if results_df_f1_era.empty: raise ValueError("No data found for F1 era (1950 onwards).")
    
    # IMPORTANT: We need prepared_data and the ID maps for the post-processing,
    # so we always need to run this part.
    prepared_data, standardization_params = prepare_data(results_df_f1_era, drivers_df, constructors_df)
    if prepared_data is None: raise ValueError("Data preparation resulted in an empty DataFrame.")
    
    # The ID maps are created *outside* the model now, so they are always available.
    driver_ids = prepared_data['driverid'].unique()
    team_ids = prepared_data['teamid'].unique()
    team_year_ids = prepared_data['team_year'].unique()
    
    # --- START OF THE NEW LOGIC ---
    
    TRACE_FILENAME = "f1_model_trace.nc"
    trace = None

    # Check if a saved trace file exists
    if os.path.exists(TRACE_FILENAME):
        print(f"\n--- Found existing trace file. Loading from {TRACE_FILENAME} ---")
        trace = az.from_netcdf(TRACE_FILENAME)
    else:
        print(f"\n--- No trace file found. Running MCMC sampling... ---")
        # Only define and run the model if the trace doesn't exist
        model, _, _, _ = create_model_year_variance_function(prepared_data.copy())
        
        FINAL_DRAWS, FINAL_TUNE = 3000, 4000
        try:
            with model:
                 trace = pm.sample(
                     draws=FINAL_DRAWS, tune=FINAL_TUNE, cores=4, random_seed=42, 
                     target_accept=0.9, progressbar=True, init="advi+adapt_diag"
                 )
                 trace.to_netcdf(TRACE_FILENAME)
                 print(f"\n--- Successfully saved MCMC trace to {TRACE_FILENAME} ---")
        except Exception as e: 
            print(f"!!! Error during sampling: {e}")

    if trace:
        print("\n--- Model Analysis and Output Generation ---")
        
        key_vars = ['beta_0', 'beta_1', 'beta_2', 'beta_3', 'beta_customer', 'beta_grid', 
                    'u_chol_corr_sigmas', 'v_chol_corr_sigmas', 'w_chol_corr_sigmas', 'sigma_e']
        summary_df = az.summary(trace, round_to=4, var_names=key_vars)
        print("\nArviZ Summary (Key Parameters):\n", summary_df)
        save_to_db(summary_df.reset_index().rename(columns={'index':'variable'}), 'model_summary', db_engine_results)

        # **CHANGE**: Updated function call to pass standardization_params.
        over_performance_df = calculate_performance_over_expectation(trace, prepared_data.copy(), driver_ids, team_ids, team_year_ids, drivers_df, standardization_params)
        if not over_performance_df.empty:
            print("\n--- Top 10 All-Time Overperformances in a Single Race ---\n", over_performance_df[['year', 'forename', 'surname', 'position', 'grid', 'performance_over_expectation']].head(10))
            save_to_db(over_performance_df, 'driver_performance_over_expectation', db_engine_results)

        # **CHANGE**: Updated function call to pass standardization_params.
        yearly_rankings_df = generate_yearly_driver_rankings(trace, prepared_data.copy(), driver_ids, standardization_params, drivers_df)
        if not yearly_rankings_df.empty:
            print("\n--- Sample Yearly 'Pure Skill' Rankings ---")
            for year_to_show in [1965, 1988, 2021]:
                if year_to_show in yearly_rankings_df['year'].values:
                    print(f"\n--- Top 5 Drivers in {year_to_show} (Pure Skill) ---\n", yearly_rankings_df[yearly_rankings_df['year'] == year_to_show].head(5)[['yearly_rank', 'forename', 'surname', 'yearly_pure_skill_score']])
            save_to_db(yearly_rankings_df, 'driver_yearly_pure_skill_rankings', db_engine_results)
        
        u0_skill_intercept = trace.posterior['u_driver_params'].mean(dim=['chain', 'draw']).values[:, 0]
        all_time_df = pd.DataFrame({'driverid': driver_ids, 'u0_skill_intercept': u0_skill_intercept})
        all_time_df = pd.merge(all_time_df, drivers_df[['driverid', 'forename', 'surname']], on='driverid').sort_values(by='u0_skill_intercept', ascending=False)
        print("\n--- Top 20 All-Time Driver Ranking (based on raw talent u0) ---\n", all_time_df[['forename', 'surname', 'u0_skill_intercept']].head(20))
        save_to_db(all_time_df, 'driver_all_time_u0_ranking', db_engine_results)

        print("\n--- Generating 2025 Head-to-Head Predictions ---")
        def parse_driver_name(slug):
            if 'jr' in slug:
                slug = slug.replace('-jr', ' Jr.')
            return slug.replace('-', ' ').title()
        
        team_pairings_2025 = [
            ("red-bull",       "max-verstappen",        "liam-lawson"),
            ("red-bull",       "max-verstappen",        "yuki-tsunoda"),
            ("ferrari",        "charles-leclerc",       "lewis-hamilton"),
            ("williams",       "carlos-sainz-jr",       "alexander-albon"),
            ("racing-bulls",   "liam-lawson",           "isack-hadjar"),
            ("racing-bulls",   "liam-lawson",           "yuki-tsunoda"),
            ("kick-sauber",    "nico-hulkenberg",       "gabriel-bortoleto"),
            ("alpine",         "pierre-gasly",          "franco-colapinto"),
            ("haas",           "esteban-ocon",          "oliver-bearman"),
            ("mercedes",       "george-russell",        "kimi-antonelli"),
            ("aston-martin",   "fernando-alonso",       "lance-stroll"),
            ("mclaren",        "oscar-piastri",         "lando-norris")
        ]
        all_h2h_results = []
        for team_id_slug, d1_slug, d2_slug in team_pairings_2025: # Use team_id_slug
            driver1_name = parse_driver_name(d1_slug)
            driver2_name = parse_driver_name(d2_slug)
            
            h2h_results = predict_teammate_head_to_head(
                trace=trace,
                driver1_name=driver1_name,
                driver2_name=driver2_name,
                team_id_slug=team_id_slug, # Pass the slug here
                future_year=2025,
                prepared_data=prepared_data,
                drivers_df=drivers_df,
                constructors_df=constructors_df,
                driver_ids_map=driver_ids,
                team_ids_map=team_ids,
                team_year_ids_map=team_year_ids,
                standardization_params=standardization_params
            )
            
            if h2h_results:
                h2h_results['constructor_id'] = team_id_slug
                h2h_results['driver1_name'] = driver1_name
                h2h_results['driver2_name'] = driver2_name
                all_h2h_results.append(h2h_results)
                
                print("\n--- Head-to-Head Prediction Result ---")
                print(f"Result for: {h2h_results['prediction_for']}")
                print(f"  > {driver1_name} outperforms: {h2h_results['prob_d1_outperforms']}")
                print(f"  > {driver2_name} outperforms: {h2h_results['prob_d2_outperforms']}")
                print(f"  > Avg Performance Gap (D1 - D2): {h2h_results['avg_performance_gap']}")
                print("--------------------------------------")
        
        if all_h2h_results:
            h2h_results_df = pd.DataFrame(all_h2h_results)
            
            # --- START OF FIX ---
            # Convert the list column to a string so the database can handle it.
            h2h_results_df['gap_94_hdi'] = h2h_results_df['gap_94_hdi'].astype(str)
            # --- END OF FIX ---

            h2h_results_df = h2h_results_df[[
                'constructor_id', 'driver1_name', 'driver2_name',
                'prob_d1_outperforms', 'prob_d2_outperforms', 
                'avg_performance_gap', 'gap_94_hdi', 'prediction_for'
            ]]
            save_to_db(h2h_results_df, 'predictions_h2h_2025', db_engine_results)
    else:
        print("\nSkipping analysis because MCMC trace is missing.")

    print("\nScript execution finished.")