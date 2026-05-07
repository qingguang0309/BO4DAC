import os
import json
import base64
from io import BytesIO
from datetime import datetime
from threading import Lock
from typing import Generator

import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template, session, redirect, Response
from flask_cors import CORS

# --- Database & Configuration ---
from database_manager import ExperimentDatabase
from config_manager import ConfigManager
from encoder import FeatureEncoder

# --- BoTorch Optimisation System ---
from optimization_system import DACOptimizer

# --- LLM-assisted Suggestions ---
from llm_service import get_llm_suggestions, stream_llm_suggestions

# --- Visualisation (matplotlib only) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Application Setup
# ----------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'dac_optimization_secret_key_2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
CORS(app)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
FEATURE_ORDER = [
    'Support',
    'Amine_1_or_Additive_1',
    'Amine_2_or_Additive_2',
    'Organic_Content_pct',
    'BET_Bare_Surface_Area_m2_g',
    'Average_Bare_Pore_Diameter_nm'
]

# Path to the master historical experiments file
HISTORICAL_CSV_PATH = 'data/historical_experiments.csv'

# Session-based BO system manager
# Each session has one BO system that gets updated with new data
session_bo_systems = {}

# ----------------------------------------------------------------------
# Global Services
# ----------------------------------------------------------------------
db = ExperimentDatabase(db_dir='data/database')
config_manager = ConfigManager('config/default_config.json')
encoder = FeatureEncoder()          # only for pre‑trained encoders (optional)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def numpy_to_python(obj):
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        # Check for NaN/Inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif pd.isna(obj):
        return None
    return obj

def get_active_session_id():
    """Return the current session ID from Flask session."""
    sid = session.get('current_session_id')
    if not sid:
        return None
    if db.get_session(sid) is None:
        session.pop('current_session_id', None)
        return None
    return sid

def create_bo_system(session_id):
    """
    Get or create the BO system for a session.
    Each session has one BO system that gets initialized once and updated with new experimental data.
    """
    # Check if we already have a BO system for this session
    if session_id in session_bo_systems:
        return session_bo_systems[session_id]

    sess = db.get_session(session_id)
    if not sess:
        raise ValueError(f"Session {session_id} not found")

    conditions = sess.get('conditions', {})
    search_bounds = sess.get('search_bounds', {})

    categorical_bounds = {
        "Support": search_bounds.get('supports', []),
        "Amine_1_or_Additive_1": search_bounds.get('amine1', []),
        "Amine_2_or_Additive_2": search_bounds.get('amine2', [])
    }

    mw_range = search_bounds.get('mwRange', [0, 10000])
    oc_range = search_bounds.get('ocRange', [0, 100])

    # Determine BET and Pore ranges based on support-specific ranges
    support_specific_ranges = search_bounds.get('supportSpecificRanges', {})
    supports_selected = search_bounds.get('supports', [])

    # Calculate overall ranges based on selected supports
    if supports_selected and support_specific_ranges:
        overall_min_bet = float('inf')
        overall_max_bet = float('-inf')
        overall_min_pore = float('inf')
        overall_max_pore = float('-inf')

        for support in supports_selected:
            if support in support_specific_ranges:
                bet_range = support_specific_ranges[support].get('betRange', [0, 1000])
                pore_range = support_specific_ranges[support].get('poreRange', [0, 20])

                overall_min_bet = min(overall_min_bet, bet_range[0])
                overall_max_bet = max(overall_max_bet, bet_range[1])
                overall_min_pore = min(overall_min_pore, pore_range[0])
                overall_max_pore = max(overall_max_pore, pore_range[1])

        # If no specific ranges were found, use default values
        if overall_min_bet == float('inf'):
            overall_min_bet = 0
        if overall_max_bet == float('-inf'):
            overall_max_bet = 1000
        if overall_min_pore == float('inf'):
            overall_min_pore = 0
        if overall_max_pore == float('-inf'):
            overall_max_pore = 20

        bet_range = [overall_min_bet, overall_max_bet]
        pore_range = [overall_min_pore, overall_max_pore]
    else:
        # Default ranges if no support-specific ranges are defined
        bet_range = [0, 1000]
        pore_range = [0, 20]

    continuous_bounds = {
        "Organic_Content_pct": (float(oc_range[0]), float(oc_range[1])),
        "BET_Bare_Surface_Area_m2_g": (float(bet_range[0]), float(bet_range[1])),
        "Average_Bare_Pore_Diameter_nm": (float(pore_range[0]), float(pore_range[1]))
    }

    bo = DACOptimizer(
        categorical_bounds=categorical_bounds,
        continuous_bounds=continuous_bounds,
        target_conditions=conditions
    )

    exps = db.get_experiments_by_session(session_id)
    if not exps:
        return bo

    X_list = []
    y_list = []
    for exp in exps:
        config = exp.get('candidate', {})
        if not config:
            continue
        # Ensure all required features are present in the config
        for feature in FEATURE_ORDER:
            if feature not in config:
                if feature in ['BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm']:
                    # Default values for new parameters - use mid-range values based on current bounds
                    if feature == 'BET_Bare_Surface_Area_m2_g':
                        # Use the mid-point of the current BET range if available
                        bet_range = bo.user_continuous_bounds.get('BET_Bare_Surface_Area_m2_g', (0, 1000))
                        config[feature] = (bet_range[0] + bet_range[1]) / 2
                    elif feature == 'Average_Bare_Pore_Diameter_nm':
                        # Use the mid-point of the current pore range if available
                        pore_range = bo.user_continuous_bounds.get('Average_Bare_Pore_Diameter_nm', (0, 20))
                        config[feature] = (pore_range[0] + pore_range[1]) / 2
                else:
                    config[feature] = config.get(feature, 0)  # Default for other parameters

        encoded = bo.encoder.encode_candidate(config, feature_order=FEATURE_ORDER)
        if encoded is not None:
            # Ensure encoded vector has the correct length
            encoded_tensor = torch.tensor(encoded, dtype=torch.float32)
            if encoded_tensor.numel() == len(FEATURE_ORDER):
                X_list.append(encoded_tensor)
                y_list.append(exp.get('experimental_performance', 0.0))

    if X_list:
        bo.train_X = torch.stack(X_list)  # Use torch.stack instead of np.array for consistency
        bo.train_Y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)
        # Fit the GP model with the historical data
        bo.fit_gp()
        app.logger.info(f"Fit records f{len(y_list)})")


    # Store the newly created BO system in the session cache
    session_bo_systems[session_id] = bo

    return bo

def add_csv_historical_data(session_id, search_bounds, conditions):
    """
    Load the master historical experiments CSV, filter rows according to the
    current search bounds and experimental conditions, and add them to the
    database as historical records.
    """
    if not os.path.exists(HISTORICAL_CSV_PATH):
        print(f"Historical CSV not found at {HISTORICAL_CSV_PATH}, skipping.")
        return

    try:
        df = pd.read_csv(HISTORICAL_CSV_PATH)
    except Exception as e:
        print(f"Error reading historical CSV: {e}")
        return

    # Mapping of CSV column names to internal field names
    col_map = {
        'Support': 'Support',
        'Amine_1_or_Additive_1': 'Amine_1_or_Additive_1',
        'Organic_Content_pct': 'Organic_Content_pct',
        'CO2_Capacity_mmol_g': 'CO2_Capacity_mmol_g',
        'Amine_2_or_Additive_2': 'Amine_2_or_Additive_2',
        'BET_Bare_Surface_Area_m2_g': 'BET_Bare_Surface_Area_m2_g',
        'Average_Bare_Pore_Diameter_nm': 'Average_Bare_Pore_Diameter_nm',
        'Adsorption_Temperature_C': 'Temperature',
        'Relative_Humidity_pct': 'Humidity',
        'CO2_Concentration_vol_pct': 'CO2_Concentration',
        'Flow_Rate_mL_min': 'Flow_Rate',
        'CO2_Test_Method': 'Test_Method'
    }

    # Tolerance for numeric comparisons
    TEMP_TOL = 1.0           # °C
    CO2_TOL = 0.1           # vol%
    HUMIDITY_TOL = 1.0      # %
    FLOW_TOL = 5.0          # mL/min

    # Pre‑process allowed sets for categorical fields
    allowed_supports = set(search_bounds.get('supports', []))
    allowed_amine1 = set(search_bounds.get('amine1', []))
    allowed_amine2 = set(search_bounds.get('amine2', []))
    allowed_amine3 = set(search_bounds.get('additive3', []))
    mw_min, mw_max = search_bounds.get('mwRange', [0, 10000])
    oc_min, oc_max = search_bounds.get('ocRange', [0, 100])
    # Determine BET and Pore ranges based on support-specific ranges for filtering
    support_specific_ranges = search_bounds.get('supportSpecificRanges', {})
    supports_selected = search_bounds.get('supports', [])

    # Calculate overall ranges based on selected supports
    if supports_selected and support_specific_ranges:
        overall_min_bet = float('inf')
        overall_max_bet = float('-inf')
        overall_min_pore = float('inf')
        overall_max_pore = float('-inf')

        for support in supports_selected:
            if support in support_specific_ranges:
                bet_range = support_specific_ranges[support].get('betRange', [0, 1000])
                pore_range = support_specific_ranges[support].get('poreRange', [0, 20])

                overall_min_bet = min(overall_min_bet, bet_range[0])
                overall_max_bet = max(overall_max_bet, bet_range[1])
                overall_min_pore = min(overall_min_pore, pore_range[0])
                overall_max_pore = max(overall_max_pore, pore_range[1])

        # If no specific ranges were found, use default values
        if overall_min_bet == float('inf'):
            overall_min_bet = 0
        if overall_max_bet == float('-inf'):
            overall_max_bet = 1000
        if overall_min_pore == float('inf'):
            overall_min_pore = 0
        if overall_max_pore == float('-inf'):
            overall_max_pore = 20

        bet_min, bet_max = overall_min_bet, overall_max_bet
        pore_min, pore_max = overall_min_pore, overall_max_pore
    else:
        # Default ranges if no support-specific ranges are defined
        bet_min, bet_max = [0, 1000]
        pore_min, pore_max = [0, 20]

    # Current conditions
    target_temp = conditions.get('temperature', 25.0)
    target_co2 = conditions.get('co2Concentration', 0.04)
    target_humidity = conditions.get('humidity', 0)
    target_flow = conditions.get('flowRate', 100.0)
    target_method = conditions.get('testMethod', 'TGA')

    best_cap = 0.0
    best_exp = None

    for idx, row in df.iterrows():
        # --- Categorical filters ---
        support = row.get('Support')
        if support not in allowed_supports:
            continue

        amine1 = row.get('Amine_1_or_Additive_1')
        if amine1 not in allowed_amine1:
            continue

        # Amine 2: CSV uses "0" for "No"
        amine2_raw = row.get('Amine_2_or_Additive_2')
        amine2 = 'No' if amine2_raw == 0 or amine2_raw == '0' or pd.isna(amine2_raw) else str(amine2_raw)
        if amine2 not in allowed_amine2:
            continue

        # --- Continuous filters ---
        oc = row.get('Organic_Content_pct')
        bet = row.get('BET_Bare_Surface_Area_m2_g')
        pore = row.get('Average_Bare_Pore_Diameter_nm')
        cap = row.get('CO2_Capacity_mmol_g')
        if pd.isna(oc) or pd.isna(cap):
            continue
        try:
            oc = float(oc)
            cap = float(cap)
            bet = float(bet) if not pd.isna(bet) else 0.0
            pore = float(pore) if not pd.isna(pore) else 0.0
        except (ValueError, TypeError):
            continue
        if oc < oc_min or oc > oc_max:
            continue
        if bet < bet_min or bet > bet_max:
            continue
        if pore < pore_min or pore > pore_max:
            continue

        # --- Condition filters ---
        temp = row.get('Adsorption_Temperature_C')
        co2_conc = row.get('CO2_Concentration_vol_pct')
        humidity = row.get('Relative_Humidity_pct')
        flow = row.get('Flow_Rate_mL_min')
        method = row.get('CO2_Test_Method')

        # Temperature
        if pd.isna(temp):
            continue
        try:
            temp = float(temp)
        except (ValueError, TypeError):
            continue
        if abs(temp - target_temp) > TEMP_TOL:
            continue

        # CO₂ concentration
        if pd.isna(co2_conc):
            continue
        try:
            co2_conc = float(co2_conc)
        except (ValueError, TypeError):
            continue
        if abs(co2_conc - target_co2) > CO2_TOL:
            continue

        # Humidity
        if pd.isna(humidity):
            continue
        try:
            humidity = float(humidity)
        except (ValueError, TypeError):
            continue
        if abs(humidity - target_humidity) > HUMIDITY_TOL:
            continue

        # Flow rate
        if pd.isna(flow):
            continue
        try:
            flow = float(flow)
        except (ValueError, TypeError):
            continue
        if abs(flow - target_flow) > FLOW_TOL:
            continue

        # Test method – exact match
        if pd.isna(method):
            continue
        if method.strip() != target_method:
            continue

        # --- Build candidate and experiment ---
        candidate = {
            'Support': support,
            'Amine_1_or_Additive_1': amine1,
            'Amine_2_or_Additive_2': amine2,
            'Organic_Content_pct': oc,
            'BET_Bare_Surface_Area_m2_g': bet,
            'Average_Bare_Pore_Diameter_nm': pore
        }

        # Ensure all required features are present
        for feature in FEATURE_ORDER:
            if feature not in candidate:
                if feature in ['BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm']:
                    candidate[feature] = 0.0
                else:
                    candidate[feature] = 0.0

        exp_data = {
            'session_id': session_id,
            'candidate': candidate,
            'predicted_performance': 0.0,
            'uncertainty': 0.0,
            'experimental_performance': cap,
            'is_historical': True,
            'timestamp': datetime.now().isoformat(),
            'Temperature': temp,
            'Humidity': humidity,
            'CO2_Concentration': co2_conc,
            'Flow_Rate': flow,
            'Test_Method': method,
            'Notes': 'Loaded from master historical CSV'
        }

        db.add_experiment(session_id, exp_data)

        if cap > best_cap:
            best_cap = cap
            best_exp = candidate

    # Update session best if we found a new record
    if best_cap > 0:
        sess = db.get_session(session_id)
        current_best = sess.get('best_capacity', 0.0)
        if best_cap > current_best:
            db.update_session(session_id, {
                'best_capacity': best_cap,
                'best_experiment': best_exp
            })

    print(f"Added rows from historical CSV, {best_cap} best capacity")

# ----------------------------------------------------------------------
# API Endpoints – 4‑Step Optimisation Flow
# ----------------------------------------------------------------------
@app.route('/api/init', methods=['POST'])
def api_init():
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    # Validate required fields
    search_bounds = data.get('searchBounds')
    conditions = data.get('conditions')
    
    if not search_bounds:
        return jsonify({'success': False, 'error': 'searchBounds are required'}), 400
    
    if not conditions:
        return jsonify({'success': False, 'error': 'conditions are required'}), 400

    # Validate that search_bounds has required sub-fields
    required_search_bounds = ['supports', 'amine1', 'amine2']
    for bound in required_search_bounds:
        if not search_bounds.get(bound):
            return jsonify({'success': False, 'error': f'{bound} is required in searchBounds'}), 400

    # Validate that conditions has required fields
    required_conditions = ['temperature', 'co2Concentration', 'humidity', 'flowRate', 'testMethod']
    for condition in required_conditions:
        if condition not in conditions or conditions[condition] is None:
            return jsonify({'success': False, 'error': f'{condition} is required in conditions'}), 400

    session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    session_data = {
        'created_at': datetime.now().isoformat(),
        'status': 'initialized',
        'search_bounds': search_bounds,
        'conditions': conditions,
        'best_capacity': 0.0,
        'best_experiment': None,
        'current_candidates': []
    }
    db.create_session(session_id, session_data)

    # 1. Add user‑provided historical records (from Step 3 form)
    for rec in data.get('historicalRecords', []):
        # Validate required fields in historical records
        if not rec.get('Support') or not rec.get('Amine_1_or_Additive_1'):
            return jsonify({'success': False, 'error': 'Support and Amine_1_or_Additive_1 are required in historical records'}), 400
            
        # Validate numeric values
        try:
            organic_content = float(rec.get('Organic_Content_pct', 0))
            bet_surface_area = float(rec.get('BET_Bare_Surface_Area_m2_g', 0))
            pore_diameter = float(rec.get('Average_Bare_Pore_Diameter_nm', 0))
            cap = float(rec.get('CO2_Capacity_mmol_g', 0))
            
            # Validate positive values where appropriate
            if organic_content < 0 or bet_surface_area < 0 or pore_diameter < 0 or cap < 0:
                return jsonify({'success': False, 'error': 'Numeric values must be non-negative'}), 400
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Numeric parameters must be valid numbers'}), 400

        candidate = {
            'Support': rec.get('Support'),
            'Amine_1_or_Additive_1': rec.get('Amine_1_or_Additive_1'),
            'Amine_2_or_Additive_2': rec.get('Amine_2_or_Additive_2', 'No'),
            'Organic_Content_pct': organic_content,
            'BET_Bare_Surface_Area_m2_g': bet_surface_area,
            'Average_Bare_Pore_Diameter_nm': pore_diameter
        }

        # Ensure all required features are present
        for feature in FEATURE_ORDER:
            if feature not in candidate:
                if feature in ['BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm']:
                    candidate[feature] = 0.0
                else:
                    candidate[feature] = 0.0

        exp_data = {
            'session_id': session_id,
            'candidate': candidate,
            'predicted_performance': 0.0,
            'uncertainty': 0.0,
            'experimental_performance': cap,
            'is_historical': True,
            'timestamp': datetime.now().isoformat()
        }
        db.add_experiment(session_id, exp_data)

    # 2. Automatically add filtered historical records from master CSV
    search_bounds = data.get('searchBounds', {})
    conditions = data.get('conditions', {})
    add_csv_historical_data(session_id, search_bounds, conditions)

    # 3. Update session best based on all records now in DB
    experiments = db.get_experiments_by_session(session_id)
    app.logger.info(f"found {len(experiments)} experiments for session {session_id}")
    best_cap = 0.0
    best_exp = None
    for exp in experiments:
        cap = exp.get('experimental_performance', 0.0)
        if cap > best_cap:
            best_cap = cap
            best_exp = exp.get('candidate')
    db.update_session(session_id, {
        'best_capacity': best_cap,
        'best_experiment': best_exp
    })

    session['current_session_id'] = session_id

    bo = create_bo_system(session_id)
    total_data_points = bo.train_X.shape[0] if bo.train_X.ndim > 0 else 0

    return jsonify({
        'success': True,
        'session_id': session_id,
        'best_capacity': best_cap,
        'data_points': total_data_points
    })

@app.route('/api/generate-candidates', methods=['POST'])
def api_generate_candidates():
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    n_candidates_input = data.get('n_candidates', 5)
    
    # Validate n_candidates is a positive integer within reasonable limits
    try:
        n_candidates = int(n_candidates_input)
        if n_candidates <= 0:
            return jsonify({'success': False, 'error': 'n_candidates must be a positive integer'}), 400
        if n_candidates > 100:  # Reasonable upper limit
            return jsonify({'success': False, 'error': 'n_candidates cannot exceed 100'}), 400
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'n_candidates must be a valid integer'}), 400

    bo = create_bo_system(sid)
    candidates = bo.generate_candidates(n_candidates)
    app.logger.info(f"genreating candidates {candidates}")
    session_data = db.get_session(sid)
    session_data['current_candidates'] = candidates
    db.update_session(sid, {'current_candidates': candidates})
    best_cap = float(bo.train_Y.max().item()) if bo.train_Y.numel() > 0 else 0.0

    return jsonify({
        'success': True,
        'candidates': numpy_to_python(candidates),
        'best_capacity': best_cap
    })


@app.route('/api/copy-session', methods=['POST'])
def api_copy_session():
    """Copy an existing session's configuration and historical records to a new session."""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data'}), 400

    source_session_id = data.get('source_session_id')
    if not source_session_id:
        return jsonify({'success': False, 'error': 'Source session ID required'}), 400

    # Get the source session
    source_session = db.get_session(source_session_id)
    if not source_session:
        return jsonify({'success': False, 'error': 'Source session not found'}), 404

    # Get source session experiments
    source_experiments = db.get_experiments_by_session(source_session_id)

    # Create new session ID
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Copy session data (excluding runtime data like best_capacity, current_candidates)
    new_session_data = {
        'created_at': datetime.now().isoformat(),
        'status': 'initialized',
        'search_bounds': source_session.get('search_bounds', {}),
        'conditions': source_session.get('conditions', {}),
        'best_capacity': 0.0,
        'best_experiment': None,
        'current_candidates': []
    }

    db.create_session(new_session_id, new_session_data)

    # Copy ALL experiments from the source session (both historical and real)
    for exp in source_experiments:
        candidate = exp.get('candidate', {})
        cap = exp.get('experimental_performance', 0.0)

        # Ensure all required features are present
        for feature in FEATURE_ORDER:
            if feature not in candidate:
                if feature in ['BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm']:
                    candidate[feature] = 0.0
                else:
                    candidate[feature] = 0.0

        exp_data = {
            'session_id': new_session_id,
            'candidate': candidate,
            'predicted_performance': exp.get('predicted_performance', 0.0),
            'uncertainty': exp.get('uncertainty', 0.0),
            'experimental_performance': cap,
            'is_historical': exp.get('is_historical', False),  # Preserve original is_historical flag
            'original_timestamp': exp.get('timestamp', datetime.now().isoformat()),  # Preserve original timestamp
            'timestamp': datetime.now().isoformat(),  # New timestamp for the copy
            'Temperature': exp.get('Temperature', 25.0),
            'Humidity': exp.get('Humidity', 0),
            'CO2_Concentration': exp.get('CO2_Concentration', 0.04),
            'Flow_Rate': exp.get('Flow_Rate', 100.0),
            'Test_Method': exp.get('Test_Method', 'TGA'),
            'Notes': exp.get('Notes', '') + f' (Copied from session {source_session_id})'
        }
        db.add_experiment(new_session_id, exp_data)

    # Update session best based on all records (both historical and real)
    experiments = db.get_experiments_by_session(new_session_id)
    best_cap = 0.0
    best_exp = None
    for exp in experiments:
        cap = exp.get('experimental_performance', 0.0)
        if cap > best_cap:
            best_cap = cap
            best_exp = exp.get('candidate')
    db.update_session(new_session_id, {
        'best_capacity': best_cap,
        'best_experiment': best_exp
    })

    # Set as current session
    session['current_session_id'] = new_session_id

    return jsonify({
        'success': True,
        'new_session_id': new_session_id,
        'message': f'Copied {len(experiments)} historical records from session {source_session_id}'
    })

@app.route('/api/set-session', methods=['POST'])
def api_set_session():
    """Set the current session ID from URL parameter."""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data'}), 400

    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID required'}), 400

    # Verify the session exists
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({'success': False, 'error': 'Session not found'}), 404

    session['current_session_id'] = session_id
    return jsonify({'success': True, 'session_id': session_id})
@app.route('/api/record-experiment', methods=['POST'])
def api_record_experiment():
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    candidate_idx = data.get('candidate_id')
    actual_capacity_input = data.get('actual_capacity')
    notes = data.get('notes', '')  # Notes can be empty

    # Validate required fields are present and not empty/None
    if candidate_idx is None or candidate_idx == '':
        return jsonify({'success': False, 'error': 'candidate_id is required and cannot be empty'}), 400

    if actual_capacity_input is None or actual_capacity_input == '':
        return jsonify({'success': False, 'error': 'actual_capacity is required and cannot be empty'}), 400

    # Validate actual_capacity is a number and positive
    try:
        actual_capacity = float(actual_capacity_input)
        if actual_capacity < 0:
            return jsonify({'success': False, 'error': 'actual_capacity must be a positive number'}), 400
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'actual_capacity must be a valid number'}), 400

    session_data = db.get_session(sid)
    candidates = session_data.get('current_candidates', [])
    if candidate_idx is None or candidate_idx == '' or candidate_idx >= len(candidates):
        return jsonify({'success': False, 'error': 'Invalid candidate index'}), 400

    candidate = candidates[candidate_idx]

    # Add to database with original uncertainty from when candidate was generated
    exp_data = {
        'session_id': sid,
        'candidate': candidate,
        'predicted_performance': candidate.get('Predicted_CO2_Capacity_mmol_g', 0.0),
        'uncertainty': candidate.get('Uncertainty', 0.0),  # Original uncertainty from when candidate was generated
        'experimental_performance': actual_capacity,
        'is_historical': False,
        'notes': notes,
        'timestamp': datetime.now().isoformat()
    }
    db.add_experiment(sid, exp_data)

    best_cap = session_data.get('best_capacity', 0.0)
    if actual_capacity > best_cap:
        db.update_session(sid, {
            'best_capacity': actual_capacity,
            'best_experiment': candidate
        })
        best_cap = actual_capacity

    # Update the existing BO system with the new experimental result
    bo = create_bo_system(sid)  # This will get the existing BO system or create a new one
    bo.add_experiment({
        **candidate,
        'actual_capacity': actual_capacity
    })

    # Update the stored BO system in the session cache
    session_bo_systems[sid] = bo

    total_data_points = bo.train_X.shape[0] if bo.train_X.ndim > 0 else 0
    real_exps = [e for e in db.get_experiments_by_session(sid) if not e.get('is_historical', False)]

    return jsonify({
        'success': True,
        'best_capacity': best_cap,
        'total_experiments': len(real_exps),
        'total_data_points': total_data_points
    })

@app.route('/api/record-experiment-full', methods=['POST'])
def api_record_experiment_full():
    """Record an experimental result with full candidate configuration including all parameters and conditions."""
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    candidate = data.get('candidate', {})
    actual_capacity_input = data.get('actual_capacity')
    notes = data.get('notes', '')  # Notes can be empty
    app.logger.info(f"submit candidate, {candidate}")
    # Validate required fields in candidate are present and not empty
    if not candidate.get('Support') or candidate.get('Support') == '':
        return jsonify({'success': False, 'error': 'Support is required and cannot be empty'}), 400
    if not candidate.get('Amine_1_or_Additive_1') or candidate.get('Amine_1_or_Additive_1') == '':
        return jsonify({'success': False, 'error': 'Amine_1_or_Additive_1 is required and cannot be empty'}), 400
    if actual_capacity_input is None or actual_capacity_input == '':
        return jsonify({'success': False, 'error': 'actual_capacity is required and cannot be empty'}), 400

    # Validate actual_capacity is a number and positive
    try:
        actual_capacity = float(actual_capacity_input)
        if actual_capacity < 0:
            return jsonify({'success': False, 'error': 'actual_capacity must be a positive number'}), 400
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'actual_capacity must be a valid number'}), 400

    # Validate required continuous parameters exist, are not empty, and are positive numbers
    required_params = [
        'Organic_Content_pct',
        'BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm'
    ]

    for param in required_params:
        param_value = candidate.get(param)
        if param_value is None or param_value == '':
            return jsonify({'success': False, 'error': f'{param} is required and cannot be empty'}), 400
        
        try:
            param_float = float(param_value)
            if param_float <= 0:
                return jsonify({'success': False, 'error': f'{param} must be a positive number'}), 400
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': f'{param} must be a valid number'}), 400

    # Get the original predicted capacity from the data or from the candidate if it's available
    original_predicted_capacity = data.get('original_predicted_capacity', candidate.get('Predicted_CO2_Capacity_mmol_g', 0.0))

    # Update the existing BO system with the new experimental result
    bo = create_bo_system(sid)  # This will get the existing BO system or create a new one
    bo.add_experiment({
        **candidate,
        'actual_capacity': actual_capacity
    })

    # Update the stored BO system in the session cache
    session_bo_systems[sid] = bo

    recalculated_uncertainty = candidate.get('Uncertainty', 0.0)

    exp_data = {
        'session_id': sid,
        'candidate': candidate,
        'predicted_performance': original_predicted_capacity,  # Use the original predicted capacity
        'uncertainty': recalculated_uncertainty,  # Recalculated using updated BO model
        'experimental_performance': actual_capacity,
        'is_historical': False,
        'notes': notes,
        'timestamp': datetime.now().isoformat(),
        # Include experimental conditions from candidate if available
        'Temperature': candidate.get('Temperature', 25.0),
        'Humidity': candidate.get('Humidity', 0),
        'CO2_Concentration': candidate.get('CO2_Concentration', 0.04),
        'Flow_Rate': candidate.get('Flow_Rate', 100.0),
        'Test_Method': candidate.get('Test_Method', 'TGA')
    }
    # app.logger.info(exp_data)
    db.add_experiment(sid, exp_data)

    session_data = db.get_session(sid)
    best_cap = session_data.get('best_capacity', 0.0)
    if actual_capacity > best_cap:
        db.update_session(sid, {
            'best_capacity': actual_capacity,
            'best_experiment': candidate
        })
        best_cap = actual_capacity

    total_data_points = bo.train_X.shape[0] if bo.train_X.ndim > 0 else 0
    real_exps = [e for e in db.get_experiments_by_session(sid) if not e.get('is_historical', False)]

    return jsonify({
        'success': True,
        'best_capacity': best_cap,
        'total_experiments': len(real_exps),
        'total_data_points': total_data_points
    })

@app.route('/api/llm-suggest', methods=['POST'])
def api_llm_suggest():
    """Stream LLM-assisted formulation suggestions via SSE."""
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    data = request.json or {}
    threshold = float(data.get('threshold', 0.5))

    session_data = db.get_session(sid)
    if not session_data:
        return jsonify({'success': False, 'error': 'Session not found'}), 404

    candidates = session_data.get('current_candidates', [])
    experiments = db.get_experiments_by_session(sid)
    search_bounds = session_data.get('search_bounds', {})
    conditions = session_data.get('conditions', {})

    # Collect optimizer configuration
    opt_config = config_manager.config.get('optimization', {})
    bo = create_bo_system(sid)
    n_observed = bo.train_X.shape[0] if bo.train_Y.numel() > 0 else 0
    best_cap = float(bo.train_Y.max().item()) if bo.train_Y.numel() > 0 else 0.0
    optimization_info = {
        'model_type': opt_config.get('default_model_type', 'gaussian_process'),
        'acquisition_function': 'qExpectedImprovement',
        'n_candidates_sampled': opt_config.get('default_n_candidates', 1000),
        'exploration_xi': opt_config.get('default_xi', 0.01),
        'confidence_level': opt_config.get('default_confidence', 0.95),
        'n_observed_data_points': n_observed,
        'best_observed_capacity': best_cap,
    }

    # Compute average uncertainty from current candidates
    if candidates:
        uncertainties = [float(c.get('Uncertainty', 0.0)) for c in candidates]
        avg_uncertainty = sum(uncertainties) / len(uncertainties)
    else:
        avg_uncertainty = 0.0

    def generate() -> Generator[str, None, None]:
        # First event: metadata
        yield f"event: meta\ndata: {json.dumps({'avg_uncertainty': avg_uncertainty, 'threshold': threshold, 'triggered': avg_uncertainty < threshold})}\n\n"

        # Stream LLM tokens
        for sse_msg in stream_llm_suggestions(
            experiments=experiments,
            search_bounds=search_bounds,
            conditions=conditions,
            candidates=candidates,
            avg_uncertainty=avg_uncertainty,
            optimization_info=optimization_info,
        ):
            yield sse_msg

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/generate-chart', methods=['GET'])
def api_generate_chart():
    """Generate optimisation progress chart – includes historical records."""
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    experiments = db.get_experiments_by_session(sid)
    if not experiments:
        # No data at all
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Optimization Progress')
    else:
        # Sort all experiments chronologically
        sorted_exps = sorted(experiments, key=lambda x: x.get('timestamp', ''))
        iterations = list(range(1, len(sorted_exps) + 1))

        actual_all = []
        predicted_real = []
        uncertainty_real = []
        iter_real = []
        is_historical_list = []

        running_best = float('-inf')
        best_sofar = []

        for i, exp in enumerate(sorted_exps):
            act = exp.get('experimental_performance', 0.0)
            actual_all.append(act)
            running_best = max(running_best, act)
            best_sofar.append(running_best)

            is_hist = exp.get('is_historical', False)
            is_historical_list.append(is_hist)

            if not is_hist:
                pred = exp.get('predicted_performance', 0.0)
                unc = exp.get('uncertainty', 0.0)
                predicted_real.append(pred)
                uncertainty_real.append(unc)
                iter_real.append(i+1)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Best capacity line
        ax.plot(iterations, best_sofar, label='Best Capacity',
                color='#27ae60', linewidth=3, marker='o')

        # 2. Actual capacities for real experiments
        real_actual = [a for a, h in zip(actual_all, is_historical_list) if not h]
        real_iter = [i for i, h in zip(iterations, is_historical_list) if not h]
        ax.plot(real_iter, real_actual, label='Real Capacity',
                color='#e74c3c', linewidth=2, linestyle='--', marker='^')

        # 3. Historical experiments
        hist_actual = [a for a, h in zip(actual_all, is_historical_list) if h]
        hist_iter = [i for i, h in zip(iterations, is_historical_list) if h]
        if hist_actual:
            ax.scatter(hist_iter, hist_actual, label='Historical Records',
                       color='#7f8c8d', marker='s', s=50, zorder=5)

        # 4. Predicted capacities - only plot if they have valid predictions from BO system
        # Exclude experiments that were manually entered without proper BO predictions (where predicted_performance is 0)
        if iter_real and predicted_real:
            # Filter out experiments with zero or invalid predictions (manually entered without BO prediction)
            valid_pred_indices = [i for i, pred in enumerate(predicted_real) if pred is not None and pred != 0.0]
            if valid_pred_indices:
                valid_pred_values = [predicted_real[i] for i in valid_pred_indices]
                valid_pred_iters = [iter_real[i] for i in valid_pred_indices]
                ax.plot(valid_pred_iters, valid_pred_values, label='Predicted Capacity',
                        color='#3498db', linewidth=2, marker='s')

        # 5. Confidence interval - only plot for experiments with valid predictions from the optimization system
        if iter_real and uncertainty_real and len(uncertainty_real) > 0:
            # Ensure uncertainty values are properly formatted
            uncertainty_real = [float(u) if u is not None else 0.0 for u in uncertainty_real]

            # Filter out any invalid uncertainty values AND only include those with valid predictions
            valid_indices = []
            for i, u in enumerate(uncertainty_real):
                has_valid_prediction = i < len(predicted_real) and predicted_real[i] is not None and predicted_real[i] != 0.0
                is_valid_uncertainty = (u is not None and
                                       not (isinstance(u, float) and (u != u or u == float('inf'))) and
                                       isinstance(u, (int, float)) and u >= 0)
                if has_valid_prediction and is_valid_uncertainty:
                    valid_indices.append(i)

            if valid_indices:
                valid_predicted = [predicted_real[i] for i in valid_indices]
                valid_uncertainty = [uncertainty_real[i] for i in valid_indices]
                valid_iter = [iter_real[i] for i in valid_indices]

                # Calculate confidence bounds using the optimization system's uncertainties
                lower_bounds = [max(0, p - 1.96 * u) for p, u in zip(valid_predicted, valid_uncertainty)]
                upper_bounds = [p + 1.96 * u for p, u in zip(valid_predicted, valid_uncertainty)]

                # Only plot if we have valid bounds
                if lower_bounds and upper_bounds:
                    ax.fill_between(valid_iter, lower_bounds, upper_bounds,
                                    color='#3498db', alpha=0.3, label='95% Confidence')

        ax.set_xlabel('Experiment (chronological)')
        ax.set_ylabel('CO₂ Capacity (mmol/g)')
        ax.set_title('Optimization Progress')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if len(iterations) > 10:
            plt.xticks(rotation=45)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return jsonify({'success': True, 'chart_data': f'data:image/png;base64,{img_base64}'})

@app.route('/api/get-status', methods=['GET'])
def api_get_status():
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    session_data = db.get_session(sid)
    experiments = db.get_experiments_by_session(sid)
    real_count = len([e for e in experiments if not e.get('is_historical', False)])
    total_data_points = len(experiments)

    # Calculate best capacity from the experiments directly, without creating BO system
    best_cap = 0.0
    best_experiment = session_data.get('best_experiment')
    for exp in experiments:
        exp_capacity = exp.get('experimental_performance', 0.0)
        if exp_capacity > best_cap:
            best_cap = exp_capacity
            best_experiment = exp.get('candidate')

    return jsonify({
        'success': True,
        'session_id': sid,
        'iteration': real_count,
        'best_capacity': best_cap,
        'best_experiment': best_experiment,
        'total_experiments': real_count,
        'total_data_points': total_data_points,
        'conditions': session_data.get('conditions', {}),
        'search_bounds': session_data.get('search_bounds', {})
    })

@app.route('/api/reset-system', methods=['POST'])
def api_reset_system():
    sid = get_active_session_id()
    if not sid:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    db.delete_session(sid)
    session.pop('current_session_id', None)
    return jsonify({'success': True, 'message': 'Session reset'})


# ----------------------------------------------------------------------
# Database Management Endpoints (unchanged)
# ----------------------------------------------------------------------
@app.route('/db/sessions', methods=['GET'])
def db_get_sessions():
    return jsonify({'success': True, 'sessions': db.get_all_sessions()})

@app.route('/db/session/<session_id>', methods=['GET'])
def db_get_session(session_id):
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    exps = db.get_experiments_by_session(session_id)
    sess['experiments'] = exps
    return jsonify({'success': True, 'session': sess})

@app.route('/db/experiments', methods=['GET'])
def db_get_experiments():
    session_id = request.args.get('session_id')
    if session_id:
        exps = db.get_experiments_by_session(session_id)
    else:
        exps = db.get_all_experiments()
    return jsonify({'success': True, 'experiments': exps, 'count': len(exps)})


@app.route('/db/export/csv', methods=['GET'])
def db_export_csv():
    """Export experiments as CSV with specific column names."""
    session_id = request.args.get('session_id')

    # Get experiments
    if session_id:
        experiments = db.get_experiments_by_session(session_id)
        if not experiments:
            return jsonify({'success': False, 'error': 'No experiments found for this session'}), 404
    else:
        experiments = db.get_all_experiments()

    if not experiments:
        return jsonify({'success': False, 'error': 'No experiments found'}), 404

    # Prepare data with specific column mapping
    export_data = []

    # Define column renaming mapping (original -> new name)
    column_rename_map = {
        # Session info
        'session_id': 'Session_ID',
        'experiment_id': 'Experiment_ID',
        'timestamp': 'Timestamp',
        'original_timestamp': 'Original_Timestamp',
        'is_historical': 'Is_Historical',

        # Performance metrics
        'predicted_performance': 'Predicted_CO2_Capacity_mmol_g',
        'experimental_performance': 'CO2_Capacity_mmol_g',
        'uncertainty': 'Uncertainty',

        # Candidate fields
        'Support': 'Support',
        'Amine_1_or_Additive_1': 'Amine_1_or_Additive_1',
        'Amine_2_or_Additive_2': 'Amine_2_or_Additive_2',
        'Organic_Content_pct': 'Organic_Content_pct',
        'BET_Bare_Surface_Area_m2_g': 'BET_Bare_Surface_Area_m2_g',
        'Average_Bare_Pore_Diameter_nm': 'Average_Bare_Pore_Diameter_nm',

        # Experimental conditions (from candidate or root)
        'Temperature': 'Temperature_C',
        'CO2_Concentration': 'CO2_Concentration_vol_pct',
        'Humidity': 'Relative_Humidity_pct',
        'Flow_Rate': 'Flow_Rate_mL_min',
        'Test_Method': 'CO2_Test_Method',

        # Notes
        'Notes': 'Notes',
        'notes': 'Notes'  # Merge both notes fields
    }

    for exp in experiments:
        row = {}
        candidate = exp.get('candidate', {})

        # Add all fields from experiment root
        for key, value in exp.items():
            if key != 'candidate':  # Skip candidate as we'll handle separately
                new_key = column_rename_map.get(key, key)
                # Handle duplicate notes field
                if new_key == 'Notes' and new_key in row:
                    # If Notes already exists, append
                    row[new_key] = str(row.get(new_key, '')) + '; ' + str(value) if value else row.get(new_key, '')
                else:
                    row[new_key] = value

        # Add fields from candidate
        for key, value in candidate.items():
            new_key = column_rename_map.get(key, key)
            # Handle potential duplicates with root fields
            if new_key in row and row[new_key] is not None and value is not None:
                # If both exist and are different, prefer candidate value for material properties
                if key in ['Support', 'Amine_1_or_Additive_1', 'Amine_2_or_Additive_2',
                           'Organic_Content_pct', 'BET_Bare_Surface_Area_m2_g',
                           'Average_Bare_Pore_Diameter_nm']:
                    row[new_key] = value
                # For conditions, prefer the values from candidate if they exist
                elif key in ['Temperature', 'CO2_Concentration', 'Humidity', 'Flow_Rate', 'Test_Method']:
                    row[new_key] = value
                elif key in ['Predicted_CO2_Capacity_mmol_g']:
                    row[new_key] = value
            else:
                row[new_key] = value


        # Ensure all required columns exist with defaults
        required_columns = [
            'Session_ID', 'Experiment_ID', 'Timestamp', 'Is_Historical',
            'Support', 'Amine_1_or_Additive_1', 'Amine_2_or_Additive_2',
            'Organic_Content_pct', 'BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm',
            'CO2_Capacity_mmol_g', 'Predicted_CO2_Capacity_mmol_g', 'Uncertainty',
            'Temperature_C', 'CO2_Concentration_vol_pct', 'Relative_Humidity_pct',
            'Flow_Rate_mL_min', 'CO2_Test_Method', 'Notes'
        ]

        for col in required_columns:
            if col not in row:
                row[col] = ''

        export_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(export_data)

    # Define the final column order
    column_order = [
        # Experiment identifiers
        'Session_ID', 'Experiment_ID', 'Timestamp', 'Original_Timestamp', 'Is_Historical',

        # Material properties
        'Support', 'Amine_1_or_Additive_1', 'Amine_2_or_Additive_2',
        'Organic_Content_pct', 'BET_Bare_Surface_Area_m2_g', 'Average_Bare_Pore_Diameter_nm',

        # Experimental conditions
        'Temperature_C', 'CO2_Concentration_vol_pct', 'Relative_Humidity_pct',
        'Flow_Rate_mL_min', 'CO2_Test_Method',

        # Performance metrics
        'CO2_Capacity_mmol_g', 'Predicted_CO2_Capacity_mmol_g', 'Uncertainty',

    ]

    # Only include columns that exist in the dataframe
    available_columns = [col for col in column_order if col in df.columns]

    # Add any remaining columns that weren't in the order (at the end)
    other_columns = [col for col in df.columns if col not in available_columns]
    final_columns = available_columns + other_columns

    df = df[final_columns]

    # Create export directory if it doesn't exist
    os.makedirs('data/export', exist_ok=True)

    # # Generate filename with timestamp
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # if session_id:
    #     filename = f'experiments_session_{session_id[:8]}_{timestamp}.csv'
    # else:
    #     filename = f'all_experiments_{timestamp}.csv'
    filename = 'experiments.csv'
    csv_file = os.path.join('data/export', filename)

    # Export to CSV with proper formatting
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    return send_file(
        csv_file,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )
@app.route('/db/export/json', methods=['GET'])
def db_export_json():
    session_id = request.args.get('session_id')
    json_str = db.export_to_json(session_id)
    os.makedirs('data/export', exist_ok=True)
    json_file = 'data/export/export.json'
    with open(json_file, 'w') as f:
        f.write(json_str)
    return send_file(json_file, as_attachment=True, download_name='experiments.json')

@app.route('/db/statistics', methods=['GET'])
def db_statistics():
    return jsonify({'success': True, 'statistics': db.get_statistics()})

@app.route('/db/backup', methods=['POST'])
def db_backup():
    success = db.backup_database('data/backups')
    return jsonify({'success': success})

@app.route('/db/reset', methods=['POST'])
def db_reset():
    success = db.reset_database()
    return jsonify({'success': success})

# ----------------------------------------------------------------------
# Configuration & Encoder Endpoints (unchanged)
# ----------------------------------------------------------------------
@app.route('/config', methods=['GET'])
def config_get():
    return jsonify({'success': True, 'config': config_manager.config})

@app.route('/config/save', methods=['POST'])
def config_save():
    path = request.json.get('path', 'config/default_config.json')
    success = config_manager.save_config(path)
    return jsonify({'success': success})

@app.route('/encoders/save', methods=['POST'])
def encoders_save():
    path = request.json.get('path', 'encoders/label_encoders.pkl')
    success = encoder.save_encoders(path)
    return jsonify({'success': success, 'path': path})

@app.route('/encoders/load', methods=['POST'])
def encoders_load():
    path = request.json.get('path')
    if not path or not os.path.exists(path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    success = encoder.load_encoders(path)
    return jsonify({'success': success})

# ----------------------------------------------------------------------
# Template Routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/experimental-form')
def experimental_form():
    return render_template('experimental_form.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/database')
def database_page():
    return render_template('database.html')

# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
@app.route('/session-detail')
def session_detail():
    """Redirect to main page with session parameter."""
    session_id = request.args.get('id')
    if session_id:
        return redirect(f'/?session={session_id}&step=4')
    else:
        return redirect('/')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'database': db.db_dir is not None})

# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/database', exist_ok=True)
    os.makedirs('data/backups', exist_ok=True)
    os.makedirs('data/export', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('encoders', exist_ok=True)

    encoder_path = config_manager.config.get('encoders_load_path', 'encoders/label_encoders.pkl') if config_manager.config else 'encoders/label_encoders.pkl'
    if os.path.exists(encoder_path):
        encoder.load_encoders(encoder_path)

    app.run(debug=True, host='0.0.0.0',port=5001,use_reloader=True)