# app.py
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import json
import pandas as pd
import torch
import numpy as np
import sys
import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import logging

# Import your optimization classes (make sure they're in the same directory)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization_system import (
    HistoricalDataProcessor,
    CatalystBOWithHistory,
    InteractiveCatalystOptimizer
)

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Global optimizer instance
optimizer = None
current_state = {
    'iteration': 0,
    'best_capacity': 0.0,
    'best_experiment': None,           # NEW: store best experiment details
    'real_experiments': [],
    'current_candidates': [],
    'conditions': {
        'Relative_Humidity_pct': 0,
        'CO2_Concentration_vol_pct': 0.04,
        'Flow_Rate_mL_min': 100.0,
        'Adsorption_Temperature_C': 25.0,
        'CO2_Test_Method': 'TGA'
    },
    'search_bounds': {}
}


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/init', methods=['POST'])
def initialize_system():
    """Initialize the optimization system"""
    global optimizer, current_state

    try:
        data = request.json
        app.logger.info(f"Initialization data received: {data}")

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Store search bounds from frontend
        if 'searchBounds' in data:
            current_state['search_bounds'] = data['searchBounds']
            app.logger.info(f"Search bounds: {current_state['search_bounds']}")
        else:
            # Default bounds
            current_state['search_bounds'] = {
                'supports': [
                'SBA-15', 'NS', 'MCM-41', 'MCM-48', 'Mesoporous γ-Al2O3', 'MMSN',
                'MCF', 'MMON', 'MPS', 'BHMS', 'SA', 'FAU', 'MIL-101(Cr)', 'MCM-36',
                'THMS', 'MF', 'NPREXAD4', 'NPRED4020', 'PREXAD7', 'PREHP2MG',
                'PREDA201', 'NPREHP20', 'R-CFA-SBA-15', 'W-CFA-SBA-15', 'ZN', 'AC',
                'FS', 'CNS', 'CA', 'PREHPD450', 'MPC'
            ],
            'amine1': [
                'BPEI', 'TEPA', 'DEA', 'DETA', 'LPEI', 'Ph-3-ED', 'Ph-3-PD',
                'Ph-6-ED', 'Ph-6-PD', 'PEG200', 'TETA', 'TPTA', 'EI-Den', 'PI-Den',
                'AM-TEPA', 'PAA', 'GPAA', 'CTMA+', 'PPG', 'LPPI', 'PGA', 'PZ',
                'MEA', 'EDA', 'Spermine', 'Spermidine', 'TREN', 'EP', 'EB-TEPA',
                'PEHA', 'AN-TEPA'
            ],
            'amine2': [
                'No', 'DEA', 'CTAB', 'P123', 'PC', 'PEG200', 'SDS', 'Span80',
                'PEG1000', 'CTAC', 'DPPD', 'TBD', 'DBPD', 'BHT', 'PET', 'TDE',
                'HEDS', 'DTDP', 'BTES', 'APTES', 'TEOT', 'CTMA+'
            ],
            'additive3': ['No', 'CTAC'],
            'mwRange': [0, 10000],
            'ocRange': [0, 100]
        }

        # Transform frontend search bounds to backend format
        categorical_bounds = {
            "Support": current_state['search_bounds'].get('supports', [
                'SBA-15', 'NS', 'MCM-41', 'MCM-48', 'Mesoporous γ-Al2O3', 'MMSN',
                'MCF', 'MMON', 'MPS', 'BHMS', 'SA', 'FAU', 'MIL-101(Cr)', 'MCM-36',
                'THMS', 'MF', 'NPREXAD4', 'NPRED4020', 'PREXAD7', 'PREHP2MG',
                'PREDA201', 'NPREHP20', 'R-CFA-SBA-15', 'W-CFA-SBA-15', 'ZN', 'AC',
                'FS', 'CNS', 'CA', 'PREHPD450', 'MPC'
            ]),
            "Amine_1_or_Additive_1": current_state['search_bounds'].get('amine1', [
                'BPEI', 'TEPA', 'DEA', 'DETA', 'LPEI', 'Ph-3-ED', 'Ph-3-PD',
                'Ph-6-ED', 'Ph-6-PD', 'PEG200', 'TETA', 'TPTA', 'EI-Den', 'PI-Den',
                'AM-TEPA', 'PAA', 'GPAA', 'CTMA+', 'PPG', 'LPPI', 'PGA', 'PZ',
                'MEA', 'EDA', 'Spermine', 'Spermidine', 'TREN', 'EP', 'EB-TEPA',
                'PEHA', 'AN-TEPA'
            ]),
            "Amine_2_or_Additive_2": current_state['search_bounds'].get('amine2', [
                'No', 'DEA', 'CTAB', 'P123', 'PC', 'PEG200', 'SDS', 'Span80',
                'PEG1000', 'CTAC', 'DPPD', 'TBD', 'DBPD', 'BHT', 'PET', 'TDE',
                'HEDS', 'DTDP', 'BTES', 'APTES', 'TEOT', 'CTMA+'
            ]),
            "Amine_3_or_Additive_3": current_state['search_bounds'].get('additive3', ['No', 'CTAC'])
        }

        continuous_bounds = {
            "MW_Mn_g_mol": (
                float(current_state['search_bounds'].get('mwRange', [0, 10000])[0]),
                float(current_state['search_bounds'].get('mwRange', [0, 10000])[1])
            ),
            "Organic_Content_pct": (
                float(current_state['search_bounds'].get('ocRange', [0, 100])[0]),
                float(current_state['search_bounds'].get('ocRange', [0, 100])[1])
            )
        }

        # Update conditions if provided
        if 'conditions' in data:
            conditions = data['conditions']
            app.logger.info(f"Conditions from frontend: {conditions}")

            condition_mapping = {
                'co2Concentration': 'CO2_Concentration_vol_pct',
                'temperature': 'Adsorption_Temperature_C',
                'humidity': 'Relative_Humidity_pct',
                'flowRate': 'Flow_Rate_mL_min',
                'testMethod': 'CO2_Test_Method'
            }

            for frontend_key, backend_key in condition_mapping.items():
                if frontend_key in conditions:
                    value = conditions[frontend_key]
                    if frontend_key == 'testMethod':
                        current_state['conditions'][backend_key] = str(value)
                    else:
                        current_state['conditions'][backend_key] = float(value)

        app.logger.info(f"Final conditions for backend: {current_state['conditions']}")

        # Initialize optimizer
        try:
            optimizer = InteractiveCatalystOptimizer(
                data_path='data/historical_experiments.csv',
                target_conditions=current_state['conditions'],
                categorical_bounds=categorical_bounds,
                continuous_bounds=continuous_bounds
            )

            # Initialize BO system
            data_processor = HistoricalDataProcessor('data/historical_experiments.csv')
            optimizer.bo_system = CatalystBOWithHistory(
                data_processor,
                current_state['conditions'],
                categorical_bounds,
                continuous_bounds
            )

            app.logger.info(f"Optimizer initialized successfully")
            app.logger.info(f"Training data size: {len(optimizer.bo_system.train_X)}")

        except Exception as e:
            app.logger.error(f"Error initializing optimizer: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to initialize optimizer: {str(e)}'
            }), 500

        # If historical records were provided, add them to the system
        if 'historicalRecords' in data and data['historicalRecords']:
            app.logger.info(f"Processing {len(data['historicalRecords'])} historical records")
            added_count = 0
            for exp in data['historicalRecords']:
                # Validate required fields
                required_fields = ['Support', 'Amine_1_or_Additive_1', 'Amine_2_or_Additive_2',
                                   'Amine_3_or_Additive_3', 'MW_Mn_g_mol', 'Organic_Content_pct', 'CO2_Capacity_mmol_g']
                if not all(field in exp for field in required_fields):
                    app.logger.warning(f"Skipping historical record due to missing required fields: {exp}")
                    continue

                config = {
                    'Support': exp['Support'],
                    'Amine_1_or_Additive_1': exp['Amine_1_or_Additive_1'],
                    'Amine_2_or_Additive_2': exp['Amine_2_or_Additive_2'],
                    'Amine_3_or_Additive_3': exp['Amine_3_or_Additive_3'],
                    'MW_Mn_g_mol': float(exp['MW_Mn_g_mol']),
                    'Organic_Content_pct': float(exp['Organic_Content_pct'])
                }
                actual_capacity = float(exp['CO2_Capacity_mmol_g'])
                optimizer.bo_system.add_experimental_result(config, actual_capacity)

                experiment_conditions = {
                    'Relative_Humidity_pct': exp.get('Humidity', current_state['conditions']['Relative_Humidity_pct']),
                    'CO2_Concentration_vol_pct': exp.get('CO2_Concentration', current_state['conditions']['CO2_Concentration_vol_pct']),
                    'Flow_Rate_mL_min': exp.get('Flow_Rate', current_state['conditions']['Flow_Rate_mL_min']),
                    'Adsorption_Temperature_C': exp.get('Temperature', current_state['conditions']['Adsorption_Temperature_C']),
                    'CO2_Test_Method': exp.get('Test_Method', current_state['conditions']['CO2_Test_Method'])
                }

                experiment_record = {
                    'id': len(current_state['real_experiments']),
                    'candidate': config,
                    'actual_capacity': actual_capacity,
                    'predicted_capacity': actual_capacity,
                    'notes': exp.get('notes', ''),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'conditions': experiment_conditions
                }
                current_state['real_experiments'].append(experiment_record)

                if actual_capacity > current_state['best_capacity']:
                    current_state['best_capacity'] = actual_capacity
                    current_state['best_experiment'] = experiment_record   # NEW

                added_count += 1

            # Limit the number of stored experiments
            MAX_EXPERIMENTS_TO_KEEP = 50
            if len(current_state['real_experiments']) > MAX_EXPERIMENTS_TO_KEEP:
                current_state['real_experiments'] = current_state['real_experiments'][-MAX_EXPERIMENTS_TO_KEEP:]

            app.logger.info(f"Successfully added {added_count} historical records")

        # Set iteration = number of real experiments
        current_state['iteration'] = len(current_state['real_experiments'])

        best_capacity = float(optimizer.bo_system.train_Y.max().item()) if len(optimizer.bo_system.train_Y) > 0 else 0.0

        return jsonify({
            'success': True,
            'message': 'System initialized successfully',
            'data_points': len(optimizer.bo_system.train_X),
            'best_capacity': best_capacity,
            'conditions': current_state['conditions'],
            'iteration': current_state['iteration']           # NEW
        })

    except Exception as e:
        app.logger.error(f"Error in initialize_system: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'System initialization failed: {str(e)}'
        }), 500


@app.route('/api/generate-candidates', methods=['POST'])
def generate_candidates():
    """Generate new catalyst candidates"""
    global optimizer, current_state

    try:
        if optimizer is None:
            app.logger.error("Optimizer not initialized")
            return jsonify({
                'success': False,
                'error': 'System not initialized. Please initialize the system first.'
            }), 400

        data = request.json or {}
        n_candidates = data.get('n_candidates', 5)

        all_candidates = optimizer.bo_system.generate_new_candidates(10)

        if not all_candidates:
            return jsonify({
                'success': True,
                'candidates': [],
                'iteration': current_state['iteration'],
                'best_capacity': current_state['best_capacity'],
                'stats': {
                    'total_generated': 0,
                    'top_predicted': 0,
                    'average_predicted': 0
                },
                'warning': 'No candidates could be generated. Please check your search bounds.'
            })

        # Sort candidates by predicted capacity
        sorted_candidates = sorted(
            all_candidates,
            key=lambda x: float(x.get('Predicted_CO2_Capacity_mmol_g', 0)),
            reverse=True
        )
        selected_candidates = sorted_candidates[:n_candidates]
        additional_candidates = sorted_candidates[n_candidates:n_candidates + 5]

        current_state['current_candidates'] = selected_candidates
        current_state['additional_candidates'] = additional_candidates
        current_state['all_generated_candidates'] = all_candidates

        # Do NOT increment iteration here – iteration = number of experiments

        # Format candidates for frontend
        formatted_candidates = []
        for i, cand in enumerate(selected_candidates):
            candidate_data = {
                'id': i,
                'Support': cand.get('Support', ''),
                'Amine_1_or_Additive_1': cand.get('Amine_1_or_Additive_1', ''),
                'Amine_2_or_Additive_2': cand.get('Amine_2_or_Additive_2', ''),
                'Amine_3_or_Additive_3': cand.get('Amine_3_or_Additive_3', ''),
                'MW_Mn_g_mol': float(cand.get('MW_Mn_g_mol', 0)),
                'Organic_Content_pct': round(float(cand.get('Organic_Content_pct', 0)), 1),
                'Predicted_CO2_Capacity_mmol_g': float(cand.get('Predicted_CO2_Capacity_mmol_g', 0)),
                'Uncertainty': float(cand.get('Uncertainty', 0)),
                'Expected_Improvement': float(cand.get('Expected_Improvement', 0)),
                'Is_New': cand.get('Is_New', True),
                'conditions': current_state['conditions'],
                'predictedCapacity': float(cand.get('Predicted_CO2_Capacity_mmol_g', 0)),
                'mw': float(cand.get('MW_Mn_g_mol', 0)),
                'organicContent': round(float(cand.get('Organic_Content_pct', 0)), 1)
            }

            # Calculate Expected Improvement if possible
            if hasattr(optimizer.bo_system, 'train_Y') and len(optimizer.bo_system.train_Y) > 0:
                best_f = optimizer.bo_system.train_Y.max().item()
                ei = max(0, candidate_data['Predicted_CO2_Capacity_mmol_g'] - best_f)
                candidate_data['Expected_Improvement'] = ei

            formatted_candidates.append(candidate_data)

        # Update best capacity from training data
        best_capacity = 0.0
        if hasattr(optimizer.bo_system, 'train_Y') and len(optimizer.bo_system.train_Y) > 0:
            best_capacity = float(optimizer.bo_system.train_Y.max().item())
        elif current_state['best_capacity'] > 0:
            best_capacity = current_state['best_capacity']

        return jsonify({
            'success': True,
            'candidates': formatted_candidates,
            'iteration': current_state['iteration'],
            'best_capacity': best_capacity,
            'stats': {
                'total_generated': len(all_candidates),
                'top_predicted': float(
                    selected_candidates[0].get('Predicted_CO2_Capacity_mmol_g', 0)) if selected_candidates else 0,
                'average_predicted': float(np.mean([c.get('Predicted_CO2_Capacity_mmol_g', 0) for c in
                                                    selected_candidates])) if selected_candidates else 0
            }
        })

    except Exception as e:
        app.logger.error(f"Error in generate_candidates: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to generate candidates: {str(e)}'
        }), 500


@app.route('/experimental-form')
def experimental_form():
    """Serve the experimental data input form"""
    try:
        return render_template('experimental_form.html')
    except Exception as e:
        app.logger.error(f"Error serving experimental form: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to load experimental form: {str(e)}'
        }), 500


@app.route('/api/record-experiment', methods=['POST'])
def record_experiment():
    """Record an experimental result from a generated candidate"""
    global optimizer, current_state

    try:
        if optimizer is None:
            return jsonify({'success': False, 'error': 'System not initialized'}), 400

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        app.logger.info(f"Recording experiment: {data}")

        if not current_state.get('current_candidates') or len(current_state['current_candidates']) == 0:
            return jsonify({'success': False, 'error': 'No candidates available to record experiment for. Please generate candidates first.'}), 400

        candidate_idx = data.get('candidate_id')
        if candidate_idx is None or candidate_idx >= len(current_state['current_candidates']):
            return jsonify({'success': False, 'error': 'Invalid candidate ID'}), 400

        candidate = current_state['current_candidates'][candidate_idx]
        app.logger.info(f"Candidate: {candidate}")

        try:
            actual_capacity = float(data.get('actual_capacity', 0))
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid capacity value'}), 400

        notes = data.get('notes', '')

        backend_candidate = {
            'Support': candidate.get('Support', ''),
            'Amine_1_or_Additive_1': candidate.get('Amine_1_or_Additive_1', ''),
            'Amine_2_or_Additive_2': candidate.get('Amine_2_or_Additive_2', ''),
            'Amine_3_or_Additive_3': candidate.get('Amine_3_or_Additive_3', ''),
            'MW_Mn_g_mol': float(candidate.get('MW_Mn_g_mol', 0)),
            'Organic_Content_pct': float(candidate.get('organicContent', candidate.get('Organic_Content_pct', 0)))
        }

        app.logger.info(f"Backend candidate: {backend_candidate}")

        original_uncertainty = float(candidate.get('Uncertainty', 0)) if candidate.get('Uncertainty') is not None else None
        original_expected_improvement = float(candidate.get('Expected_Improvement', 0)) if candidate.get('Expected_Improvement') is not None else None
        original_predicted_capacity = float(candidate.get('predictedCapacity', candidate.get('Predicted_CO2_Capacity_mmol_g', 0))) if (candidate.get('predictedCapacity') is not None or candidate.get('Predicted_CO2_Capacity_mmol_g') is not None) else None

        optimizer.bo_system.add_experimental_result(backend_candidate, actual_capacity,
                                                   original_uncertainty=original_uncertainty,
                                                   original_expected_improvement=original_expected_improvement,
                                                   original_predicted_capacity=original_predicted_capacity)

        experiment_conditions = current_state['conditions'].copy()
        experiment_record = {
            'id': len(current_state['real_experiments']),
            'candidate': backend_candidate,
            'actual_capacity': actual_capacity,
            'predicted_capacity': float(candidate.get('predictedCapacity', candidate.get('Predicted_CO2_Capacity_mmol_g', 0))),
            'notes': notes,
            'timestamp': pd.Timestamp.now().isoformat(),
            'conditions': experiment_conditions
        }

        current_state['real_experiments'].append(experiment_record)
        current_state['iteration'] = len(current_state['real_experiments'])   # update iteration

        if actual_capacity > current_state['best_capacity']:
            current_state['best_capacity'] = actual_capacity
            current_state['best_experiment'] = experiment_record

        MAX_EXPERIMENTS_TO_KEEP = 50
        if len(current_state['real_experiments']) > MAX_EXPERIMENTS_TO_KEEP:
            current_state['real_experiments'] = current_state['real_experiments'][-MAX_EXPERIMENTS_TO_KEEP:]

        # Get updated total data points
        total_data_points = len(optimizer.bo_system.train_X) if hasattr(optimizer.bo_system, 'train_X') else 0

        return jsonify({
            'success': True,
            'experiment_id': experiment_record['id'],
            'best_capacity': current_state['best_capacity'],
            'total_experiments': len(current_state['real_experiments']),
            'iteration': current_state['iteration'],
            'total_data_points': total_data_points
        })

    except Exception as e:
        app.logger.error(f"Error in record_experiment: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to record experiment: {str(e)}'
        }), 500


@app.route('/api/input-historical-records', methods=['POST'])
def input_historical_records():
    """Input historical experimental records to initialize the optimization model"""
    global optimizer, current_state

    try:
        data = request.json
        if not data or 'experiments' not in data:
            return jsonify({'success': False, 'error': 'No historical data provided'}), 400

        experiments = data['experiments']
        app.logger.info(f"Inputting {len(experiments)} historical data points")

        # If the system isn't initialized, initialize it with default parameters and the provided data
        if optimizer is None:
            app.logger.info("System not initialized, initializing with default parameters and historical data")

            # Set default conditions
            current_state['conditions'] = {
                'Relative_Humidity_pct': 0,
                'CO2_Concentration_vol_pct': 0.04,
                'Flow_Rate_mL_min': 100.0,
                'Adsorption_Temperature_C': 25.0,
                'CO2_Test_Method': 'TGA'
            }

            # Set default search bounds
            current_state['search_bounds'] = {
                'supports': [
                    'SBA-15', 'NS', 'MCM-41', 'MCM-48', 'Mesoporous γ-Al2O3', 'MMSN',
                    'MCF', 'MMON', 'MPS', 'BHMS', 'SA', 'FAU', 'MIL-101(Cr)', 'MCM-36',
                    'THMS', 'MF', 'NPREXAD4', 'NPRED4020', 'PREXAD7', 'PREHP2MG',
                    'PREDA201', 'NPREHP20', 'R-CFA-SBA-15', 'W-CFA-SBA-15', 'ZN', 'AC',
                    'FS', 'CNS', 'CA', 'PREHPD450', 'MPC'
                ],
                'amine1': [
                    'BPEI', 'TEPA', 'DEA', 'DETA', 'LPEI', 'Ph-3-ED', 'Ph-3-PD',
                    'Ph-6-ED', 'Ph-6-PD', 'PEG200', 'TETA', 'TPTA', 'EI-Den', 'PI-Den',
                    'AM-TEPA', 'PAA', 'GPAA', 'CTMA+', 'PPG', 'LPPI', 'PGA', 'PZ',
                    'MEA', 'EDA', 'Spermine', 'Spermidine', 'TREN', 'EP', 'EB-TEPA',
                    'PEHA', 'AN-TEPA'
                ],
                'amine2': [
                    'No', 'DEA', 'CTAB', 'P123', 'PC', 'PEG200', 'SDS', 'Span80',
                    'PEG1000', 'CTAC', 'DPPD', 'TBD', 'DBPD', 'BHT', 'PET', 'TDE',
                    'HEDS', 'DTDP', 'BTES', 'APTES', 'TEOT', 'CTMA+'
                ],
                'additive3': ['No', 'CTAC'],
                'mwRange': [0, 10000],
                'ocRange': [0, 100]
            }

            categorical_bounds = {
                "Support": current_state['search_bounds'].get('supports', []),
                "Amine_1_or_Additive_1": current_state['search_bounds'].get('amine1', []),
                "Amine_2_or_Additive_2": current_state['search_bounds'].get('amine2', []),
                "Amine_3_or_Additive_3": current_state['search_bounds'].get('additive3', [])
            }

            continuous_bounds = {
                "MW_Mn_g_mol": (
                    float(current_state['search_bounds'].get('mwRange', [0, 10000])[0]),
                    float(current_state['search_bounds'].get('mwRange', [0, 10000])[1])
                ),
                "Organic_Content_pct": (
                    float(current_state['search_bounds'].get('ocRange', [0, 100])[0]),
                    float(current_state['search_bounds'].get('ocRange', [0, 100])[1])
                )
            }

            try:
                data_processor = HistoricalDataProcessor('data/historical_experiments.csv')
                bo_system = CatalystBOWithHistory(
                    data_processor,
                    current_state['conditions'],
                    categorical_bounds,
                    continuous_bounds
                )

                optimizer = InteractiveCatalystOptimizer(
                    data_path='data/historical_experiments.csv',
                    target_conditions=current_state['conditions'],
                    categorical_bounds=categorical_bounds,
                    continuous_bounds=continuous_bounds
                )
                optimizer.bo_system = bo_system
                app.logger.info(f"Optimizer initialized successfully with historical data")
            except Exception as e:
                app.logger.error(f"Error initializing optimizer: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to initialize optimizer: {str(e)}'
                }), 500

        added_count = 0
        for exp in experiments:
            required_fields = ['Support', 'Amine_1_or_Additive_1', 'Amine_2_or_Additive_2',
                               'Amine_3_or_Additive_3', 'MW_Mn_g_mol', 'Organic_Content_pct', 'CO2_Capacity_mmol_g']
            if not all(field in exp for field in required_fields):
                app.logger.warning(f"Skipping experiment due to missing required fields: {exp}")
                continue

            config = {
                'Support': exp['Support'],
                'Amine_1_or_Additive_1': exp['Amine_1_or_Additive_1'],
                'Amine_2_or_Additive_2': exp['Amine_2_or_Additive_2'],
                'Amine_3_or_Additive_3': exp['Amine_3_or_Additive_3'],
                'MW_Mn_g_mol': float(exp['MW_Mn_g_mol']),
                'Organic_Content_pct': float(exp['Organic_Content_pct'])
            }
            actual_capacity = float(exp['CO2_Capacity_mmol_g'])

            optimizer.bo_system.add_experimental_result(config, actual_capacity)

            experiment_conditions = {
                'Relative_Humidity_pct': exp.get('Humidity', current_state['conditions']['Relative_Humidity_pct']),
                'CO2_Concentration_vol_pct': exp.get('CO2_Concentration', current_state['conditions']['CO2_Concentration_vol_pct']),
                'Flow_Rate_mL_min': exp.get('Flow_Rate', current_state['conditions']['Flow_Rate_mL_min']),
                'Adsorption_Temperature_C': exp.get('Temperature', current_state['conditions']['Adsorption_Temperature_C']),
                'CO2_Test_Method': exp.get('Test_Method', current_state['conditions']['CO2_Test_Method'])
            }

            experiment_record = {
                'id': len(current_state['real_experiments']),
                'candidate': config,
                'actual_capacity': actual_capacity,
                'predicted_capacity': actual_capacity,
                'notes': exp.get('notes', ''),
                'timestamp': pd.Timestamp.now().isoformat(),
                'conditions': experiment_conditions
            }
            current_state['real_experiments'].append(experiment_record)

            if actual_capacity > current_state['best_capacity']:
                current_state['best_capacity'] = actual_capacity
                current_state['best_experiment'] = experiment_record

            added_count += 1

        current_state['iteration'] = len(current_state['real_experiments'])

        MAX_EXPERIMENTS_TO_KEEP = 50
        if len(current_state['real_experiments']) > MAX_EXPERIMENTS_TO_KEEP:
            current_state['real_experiments'] = current_state['real_experiments'][-MAX_EXPERIMENTS_TO_KEEP:]

        app.logger.info(f"Successfully added {added_count} historical data points as training data")

        return jsonify({
            'success': True,
            'added_count': added_count,
            'total_experiments': len(current_state['real_experiments']),
            'best_capacity': current_state['best_capacity'],
            'total_data_points': len(optimizer.bo_system.train_X),
            'message': f'Successfully added {added_count} historical records to the optimization model as training data'
        })

    except Exception as e:
        app.logger.error(f"Error in input_historical_records: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to input historical records: {str(e)}'
        }), 500


@app.route('/api/input-custom-experimental-data', methods=['POST'])
def input_custom_experimental_data():
    """Input custom experimental data while maintaining the same conditions and search space"""
    global optimizer, current_state

    try:
        if optimizer is None:
            return jsonify({'success': False, 'error': 'System not initialized'}), 400

        data = request.json
        if not data or 'experiments' not in data:
            return jsonify({'success': False, 'error': 'No experimental data provided'}), 400

        experiments = data['experiments']
        app.logger.info(f"Inputting {len(experiments)} custom experimental data points")

        added_count = 0
        for exp in experiments:
            required_fields = ['Support', 'Amine_1_or_Additive_1', 'Amine_2_or_Additive_2',
                               'Amine_3_or_Additive_3', 'MW_Mn_g_mol', 'Organic_Content_pct', 'CO2_Capacity_mmol_g']
            if not all(field in exp for field in required_fields):
                app.logger.warning(f"Skipping experiment due to missing required fields: {exp}")
                continue

            config = {
                'Support': exp['Support'],
                'Amine_1_or_Additive_1': exp['Amine_1_or_Additive_1'],
                'Amine_2_or_Additive_2': exp['Amine_2_or_Additive_2'],
                'Amine_3_or_Additive_3': exp['Amine_3_or_Additive_3'],
                'MW_Mn_g_mol': float(exp['MW_Mn_g_mol']),
                'Organic_Content_pct': float(exp['Organic_Content_pct'])
            }
            actual_capacity = float(exp['CO2_Capacity_mmol_g'])

            optimizer.bo_system.add_experimental_result(config, actual_capacity)

            experiment_conditions = {
                'Relative_Humidity_pct': exp.get('Humidity', current_state['conditions']['Relative_Humidity_pct']),
                'CO2_Concentration_vol_pct': exp.get('CO2_Concentration', current_state['conditions']['CO2_Concentration_vol_pct']),
                'Flow_Rate_mL_min': exp.get('Flow_Rate', current_state['conditions']['Flow_Rate_mL_min']),
                'Adsorption_Temperature_C': exp.get('Temperature', current_state['conditions']['Adsorption_Temperature_C']),
                'CO2_Test_Method': exp.get('Test_Method', current_state['conditions']['CO2_Test_Method'])
            }

            experiment_record = {
                'id': len(current_state['real_experiments']),
                'candidate': config,
                'actual_capacity': actual_capacity,
                'predicted_capacity': actual_capacity,
                'notes': exp.get('notes', ''),
                'timestamp': pd.Timestamp.now().isoformat(),
                'conditions': experiment_conditions
            }
            current_state['real_experiments'].append(experiment_record)

            if actual_capacity > current_state['best_capacity']:
                current_state['best_capacity'] = actual_capacity
                current_state['best_experiment'] = experiment_record

            added_count += 1

        current_state['iteration'] = len(current_state['real_experiments'])

        MAX_EXPERIMENTS_TO_KEEP = 50
        if len(current_state['real_experiments']) > MAX_EXPERIMENTS_TO_KEEP:
            current_state['real_experiments'] = current_state['real_experiments'][-MAX_EXPERIMENTS_TO_KEEP:]

        app.logger.info(f"Successfully added {added_count} custom experimental data points")

        return jsonify({
            'success': True,
            'added_count': added_count,
            'total_experiments': len(current_state['real_experiments']),
            'best_capacity': current_state['best_capacity'],
            'iteration': current_state['iteration'],
            'total_data_points': len(optimizer.bo_system.train_X),
            'message': f'Successfully added {added_count} custom experimental data points'
        })

    except Exception as e:
        app.logger.error(f"Error in input_custom_experimental_data: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to input custom experimental data: {str(e)}'
        }), 500


@app.route('/api/input-custom-experimental-results', methods=['POST'])
def input_experimental_results():
    """Alias for input_custom_experimental_data (used by experimental_form.html)"""
    return input_custom_experimental_data()


@app.route('/api/get-history', methods=['GET'])
def get_history():
    """Get experiment history"""
    try:
        experiments = current_state['real_experiments'][-10:]
        return jsonify({
            'success': True,
            'experiments': experiments,
            'total': len(current_state['real_experiments'])
        })
    except Exception as e:
        app.logger.error(f"Error in get_history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/get-status', methods=['GET'])
def get_status():
    """Get current system status"""
    try:
        if optimizer is None:
            return jsonify({
                'success': True,
                'initialized': False,
                'message': 'System not initialized',
                'iteration': current_state['iteration'],
                'best_capacity': current_state['best_capacity'],
                'best_experiment': current_state['best_experiment'],
                'total_experiments': len(current_state['real_experiments']),
                'total_data_points': 0,
                'data_mode': 'not initialized',
                'conditions': current_state['conditions'],
                'search_bounds': current_state.get('search_bounds', {}),
                'bounds': {}
            })

        total_data_points = len(optimizer.bo_system.train_X) if hasattr(optimizer.bo_system, 'train_X') else 0

        bounds_info = {}
        if hasattr(optimizer, 'bo_system') and optimizer.bo_system:
            bo_system = optimizer.bo_system
            bounds_info = {
                'supports': getattr(bo_system, 'supports', []),
                'amine1': getattr(bo_system, 'amine1', []),
                'amine2': getattr(bo_system, 'amine2', []),
                'additive3': getattr(bo_system, 'additive3', []),
                'mwRange': getattr(bo_system, 'mw_range', [0, 20000]),
                'ocRange': getattr(bo_system, 'oc_range', [0, 100])
            }

            if not bounds_info['supports']:
                bounds_info['supports'] = current_state.get('search_bounds', {}).get('supports', [])
            if not bounds_info['amine1']:
                bounds_info['amine1'] = current_state.get('search_bounds', {}).get('amine1', [])
            if not bounds_info['amine2']:
                bounds_info['amine2'] = current_state.get('search_bounds', {}).get('amine2', [])
            if not bounds_info['additive3']:
                bounds_info['additive3'] = current_state.get('search_bounds', {}).get('additive3', [])
            if bounds_info['mwRange'] == [0, 20000]:
                bounds_info['mwRange'] = current_state.get('search_bounds', {}).get('mwRange', [0, 20000])
            if bounds_info['ocRange'] == [0, 100]:
                bounds_info['ocRange'] = current_state.get('search_bounds', {}).get('ocRange', [0, 100])

        return jsonify({
            'success': True,
            'initialized': True,
            'iteration': current_state['iteration'],
            'best_capacity': current_state['best_capacity'],
            'best_experiment': current_state['best_experiment'],
            'total_experiments': len(current_state['real_experiments']),
            'total_data_points': total_data_points,
            'data_mode': 'real-only' if optimizer.bo_system.use_real_data_only else 'combined',
            'conditions': current_state['conditions'],
            'search_bounds': current_state.get('search_bounds', {}),
            'bounds': bounds_info
        })

    except Exception as e:
        app.logger.error(f"Error in get_status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/toggle-data-mode', methods=['POST'])
def toggle_data_mode():
    """Toggle between data modes (no minimum sample requirement)"""
    global optimizer

    try:
        if optimizer is None:
            return jsonify({'success': False, 'error': 'System not initialized'}), 400

        data = request.json or {}
        mode = data.get('mode')

        if mode == 'real-only':
            optimizer.bo_system.use_real_data_only = True
        else:
            optimizer.bo_system.use_real_data_only = False

        app.logger.info(f"Data mode toggled to: {'real-only' if optimizer.bo_system.use_real_data_only else 'combined'}")

        return jsonify({
            'success': True,
            'mode': 'real-only' if optimizer.bo_system.use_real_data_only else 'combined'
        })

    except Exception as e:
        app.logger.error(f"Error in toggle_data_mode: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export-data', methods=['GET'])
def export_data():
    """Export all data as CSV"""
    global optimizer, current_state

    try:
        data = {
            'metadata': {
                'export_date': pd.Timestamp.now().isoformat(),
                'iteration': current_state['iteration'],
                'best_capacity': current_state['best_capacity']
            },
            'experiments': current_state['real_experiments'],
            'current_conditions': current_state['conditions'],
            'search_bounds': current_state.get('search_bounds', {})
        }

        json_data = json.dumps(data, indent=2, default=str)
        buffer = BytesIO()
        buffer.write(json_data.encode('utf-8'))
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'dac_optimization_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

    except Exception as e:
        app.logger.error(f"Error in export_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/update-conditions', methods=['POST'])
def update_conditions():
    """Update experimental conditions"""
    global current_state, optimizer

    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        app.logger.info(f"Updating conditions: {data}")

        condition_mapping = {
            'co2Concentration': 'CO2_Concentration_vol_pct',
            'temperature': 'Adsorption_Temperature_C',
            'humidity': 'Relative_Humidity_pct',
            'flowRate': 'Flow_Rate_mL_min',
            'testMethod': 'CO2_Test_Method'
        }

        for frontend_key, value in data.items():
            if frontend_key in condition_mapping:
                backend_key = condition_mapping[frontend_key]
                if frontend_key == 'testMethod':
                    current_state['conditions'][backend_key] = str(value)
                else:
                    current_state['conditions'][backend_key] = float(value)

        if optimizer is not None:
            optimizer.target_conditions = current_state['conditions']
            optimizer.bo_system.target_conditions = current_state['conditions']
            app.logger.info("Conditions updated in optimizer")

        return jsonify({
            'success': True,
            'conditions': current_state['conditions']
        })

    except Exception as e:
        app.logger.error(f"Error in update_conditions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/get-optimization-history', methods=['GET'])
def get_optimization_history():
    """Get optimization progress history for visualization"""
    global optimizer

    try:
        if optimizer is None or optimizer.bo_system is None:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 400

        history = optimizer.bo_system.get_optimization_history()

        return jsonify({
            'success': True,
            'history': history,
            'total_entries': len(history)
        })

    except Exception as e:
        app.logger.error(f"Error in get_optimization_history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/update-config', methods=['POST'])
def update_config():
    """Update the optimizer configuration with new bounds and conditions"""
    global optimizer, current_state

    try:
        if optimizer is None:
            return jsonify({'success': False, 'error': 'System not initialized'}), 400

        data = request.json
        app.logger.info(f"Updating configuration: {data}")

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        if 'searchBounds' in data:
            current_state['search_bounds'] = data['searchBounds']
            app.logger.info(f"Updated search bounds: {current_state['search_bounds']}")

        if 'conditions' in data:
            conditions = data['conditions']
            condition_mapping = {
                'co2Concentration': 'CO2_Concentration_vol_pct',
                'temperature': 'Adsorption_Temperature_C',
                'humidity': 'Relative_Humidity_pct',
                'flowRate': 'Flow_Rate_mL_min',
                'testMethod': 'CO2_Test_Method'
            }
            for frontend_key, backend_key in condition_mapping.items():
                if frontend_key in conditions:
                    value = conditions[frontend_key]
                    if frontend_key == 'testMethod':
                        current_state['conditions'][backend_key] = str(value)
                    else:
                        current_state['conditions'][backend_key] = float(value)

        categorical_bounds = {
            "Support": current_state['search_bounds'].get('supports', []),
            "Amine_1_or_Additive_1": current_state['search_bounds'].get('amine1', []),
            "Amine_2_or_Additive_2": current_state['search_bounds'].get('amine2', []),
            "Amine_3_or_Additive_3": current_state['search_bounds'].get('additive3', [])
        }

        continuous_bounds = {
            "MW_Mn_g_mol": (
                float(current_state['search_bounds'].get('mwRange', [0, 10000])[0]),
                float(current_state['search_bounds'].get('mwRange', [0, 10000])[1])
            ),
            "Organic_Content_pct": (
                float(current_state['search_bounds'].get('ocRange', [0, 100])[0]),
                float(current_state['search_bounds'].get('ocRange', [0, 100])[1])
            )
        }

        optimizer.bo_system.update_bounds_and_conditions(
            current_state['conditions'],
            categorical_bounds,
            continuous_bounds
        )

        app.logger.info("Configuration updated successfully")
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully',
            'conditions': current_state['conditions'],
            'search_bounds': current_state['search_bounds']
        })

    except Exception as e:
        app.logger.error(f"Error updating configuration: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Configuration update failed: {str(e)}'
        }), 500


@app.route('/api/archive-experiments', methods=['POST'])
def archive_experiments():
    """Manually trigger archiving of old experiments"""
    global optimizer, current_state

    if optimizer is None:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400

    try:
        optimizer.bo_system.archive_old_experiments(keep_last_n=50)
        optimizer.bo_system.archive_old_history(keep_last_n=100)

        MAX_EXPERIMENTS_TO_KEEP = 50
        if len(current_state['real_experiments']) > MAX_EXPERIMENTS_TO_KEEP:
            current_state['real_experiments'] = current_state['real_experiments'][-MAX_EXPERIMENTS_TO_KEEP:]

        app.logger.info("Manual archiving completed")
        return jsonify({
            'success': True,
            'experiments_remaining': len(current_state['real_experiments']),
            'history_remaining': len(optimizer.bo_system.get_optimization_history())
        })
    except Exception as e:
        app.logger.error(f"Error during archiving: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-chart', methods=['GET'])
def generate_chart():
    """Generate optimization chart as image, including historical data as separate points"""
    global optimizer

    try:
        if optimizer is None or optimizer.bo_system is None:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 400

        fig, ax = plt.subplots(figsize=(10, 6))

        # --- Historical data points (from initial CSV, if any) ---
        historical_df = optimizer.bo_system.historical_df
        if historical_df is not None and len(historical_df) > 0:
            # Filter to only rows with valid capacity
            hist_capacities = historical_df['CO2_Capacity_mmol_g'].dropna()
            if len(hist_capacities) > 0:
                # Plot at iteration 0, with transparency and distinct marker
                ax.scatter([0] * len(hist_capacities), hist_capacities,
                           color='gray', alpha=0.6, marker='x', s=60, label='Historical Data')

        # --- Real experiment history (optimization iterations) ---
        history = optimizer.bo_system.get_optimization_history()
        if history:
            iterations = [entry['iteration'] for entry in history]
            actual_capacities = [entry['actual_capacity'] for entry in history]
            predicted_capacities = [entry['predicted_capacity'] for entry in history]
            uncertainties = [entry['uncertainty'] for entry in history]

            best_capacities = []
            running_best = float('-inf')
            for actual in actual_capacities:
                running_best = max(running_best, actual)
                best_capacities.append(running_best)

            upper_bounds = [pred + 1.96 * unc for pred, unc in zip(predicted_capacities, uncertainties)]
            lower_bounds = [max(0, pred - 1.96 * unc) for pred, unc in zip(predicted_capacities, uncertainties)]

            ax.plot(iterations, best_capacities, label='Best Capacity', color='#27ae60', linewidth=3, marker='o')
            ax.plot(iterations, predicted_capacities, label='Predicted Capacity', color='#3498db', linewidth=2, marker='s')
            ax.plot(iterations, actual_capacities, label='Real Capacity', color='#e74c3c', linewidth=2, linestyle='--', marker='^')
            ax.fill_between(iterations, lower_bounds, upper_bounds, color='#3498db', alpha=0.3, label='Confidence Interval')

            ax.set_xlabel('Iteration')
            ax.set_ylabel('CO₂ Capacity (mmol/g)')
            ax.set_title('Optimization Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if len(iterations) > 10:
                plt.xticks(rotation=45)
        else:
            ax.text(0.5, 0.5, 'No experimental data yet', horizontalalignment='center',
                   verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Optimization Progress')

        img_buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)

        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)

        return jsonify({
            'success': True,
            'chart_data': f"data:image/png;base64,{img_base64}"
        })

    except Exception as e:
        app.logger.error(f"Error generating chart: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({
        'success': True,
        'message': 'API is working',
        'optimizer_initialized': optimizer is not None,
        'current_state': {
            'iteration': current_state['iteration'],
            'best_capacity': current_state['best_capacity'],
            'experiments_count': len(current_state['real_experiments'])
        }
    })


@app.route('/api/reset-system', methods=['POST'])
def reset_system():
    """Reset the optimization system while preserving the current conditions and search space"""
    global optimizer, current_state

    try:
        current_conditions = current_state['conditions'].copy()
        current_search_bounds = current_state['search_bounds'].copy()

        current_state = {
            'iteration': 0,
            'best_capacity': 0.0,
            'best_experiment': None,
            'real_experiments': [],
            'current_candidates': [],
            'conditions': current_conditions,
            'search_bounds': current_search_bounds
        }

        try:
            categorical_bounds = {
                "Support": current_state['search_bounds'].get('supports', [
                    'SBA-15', 'NS', 'MCM-41', 'MCM-48', 'Mesoporous γ-Al2O3', 'MMSN',
                    'MCF', 'MMON', 'MPS', 'BHMS', 'SA', 'FAU', 'MIL-101(Cr)', 'MCM-36',
                    'THMS', 'MF', 'NPREXAD4', 'NPRED4020', 'PREXAD7', 'PREHP2MG',
                    'PREDA201', 'NPREHP20', 'R-CFA-SBA-15', 'W-CFA-SBA-15', 'ZN', 'AC',
                    'FS', 'CNS', 'CA', 'PREHPD450', 'MPC'
                ]),
                "Amine_1_or_Additive_1": current_state['search_bounds'].get('amine1', [
                    'BPEI', 'TEPA', 'DEA', 'DETA', 'LPEI', 'Ph-3-ED', 'Ph-3-PD',
                    'Ph-6-ED', 'Ph-6-PD', 'PEG200', 'TETA', 'TPTA', 'EI-Den', 'PI-Den',
                    'AM-TEPA', 'PAA', 'GPAA', 'CTMA+', 'PPG', 'LPPI', 'PGA', 'PZ',
                    'MEA', 'EDA', 'Spermine', 'Spermidine', 'TREN', 'EP', 'EB-TEPA',
                    'PEHA', 'AN-TEPA'
                ]),
                "Amine_2_or_Additive_2": current_state['search_bounds'].get('amine2', [
                    'No', 'DEA', 'CTAB', 'P123', 'PC', 'PEG200', 'SDS', 'Span80',
                    'PEG1000', 'CTAC', 'DPPD', 'TBD', 'DBPD', 'BHT', 'PET', 'TDE',
                    'HEDS', 'DTDP', 'BTES', 'APTES', 'TEOT', 'CTMA+'
                ]),
                "Amine_3_or_Additive_3": current_state['search_bounds'].get('additive3', ['No', 'CTAC'])
            }

            continuous_bounds = {
                "MW_Mn_g_mol": (
                    float(current_state['search_bounds'].get('mwRange', [0, 10000])[0]),
                    float(current_state['search_bounds'].get('mwRange', [0, 10000])[1])
                ),
                "Organic_Content_pct": (
                    float(current_state['search_bounds'].get('ocRange', [0, 100])[0]),
                    float(current_state['search_bounds'].get('ocRange', [0, 100])[1])
                )
            }

            optimizer = InteractiveCatalystOptimizer(
                data_path='data/historical_experiments.csv',
                target_conditions=current_state['conditions'],
                categorical_bounds=categorical_bounds,
                continuous_bounds=continuous_bounds
            )

            data_processor = HistoricalDataProcessor('data/historical_experiments.csv')
            optimizer.bo_system = CatalystBOWithHistory(
                data_processor,
                current_state['conditions'],
                categorical_bounds,
                continuous_bounds
            )

            app.logger.info(f"Optimizer re-initialized successfully with preserved conditions and search bounds")
        except Exception as e:
            app.logger.error(f"Error re-initializing optimizer: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to re-initialize optimizer: {str(e)}'
            }), 500

        return jsonify({
            'success': True,
            'message': 'System reset successfully while preserving conditions and search bounds',
            'conditions': current_state['conditions'],
            'search_bounds': current_state['search_bounds']
        })

    except Exception as e:
        app.logger.error(f"Error in reset_system: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'System reset failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs('data', exist_ok=True)

    app.run(debug=True, port=5001, use_reloader=True)