import math
import os
import json
import hashlib
import pickle
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import List, Dict, Optional, Any, Tuple
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

from encoder import FeatureEncoder


class History:
    """
    Stores the optimisation history: each experiment added is recorded
    with its actual result and the predictions that were made before it.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add_entry(self, config: Dict, actual_capacity: float,
                  predicted_capacity: Optional[float] = None,
                  uncertainty: Optional[float] = None,
                  expected_improvement: Optional[float] = None) -> None:
        """
        Append one experimental result to the history.
        """
        entry = {
            'iteration': len(self.entries) + 1,
            'config': config.copy(),
            'actual_capacity': actual_capacity,
            'predicted_capacity': predicted_capacity,
            'uncertainty': uncertainty,
            'expected_improvement': expected_improvement,
        }
        self.entries.append(entry)

    def get_best_so_far(self) -> float:
        """Return the maximum actual CO₂ capacity seen so far."""
        if not self.entries:
            return -float('inf')
        return max(e['actual_capacity'] for e in self.entries)

    def get_all(self) -> List[Dict]:
        """Return the complete history list."""
        return self.entries


class DACOptimizer:
    """
    Bayesian Optimisation core using BoTorch. Handles surrogate model,
    acquisition function optimisation, and candidate generation.
    """
    FEATURE_NAMES = [
        'Support',
        'Amine_1_or_Additive_1',
        'Amine_2_or_Additive_2',
        'Organic_Content_pct',
        'BET_Bare_Surface_Area_m2_g',
        'Average_Bare_Pore_Diameter_nm',
        'Temperature',
        'CO2_Concentration',
        'Humidity',
        'Flow_Rate'
    ]
    # Indices for categorical / continuous features (0‑based)
    CATEGORICAL_DIMS = [0, 1, 2]
    CONTINUOUS_DIMS = [3, 4, 5, 6, 7, 8, 9]
    Q_VALUES = 10
    def __init__(self,
                 categorical_bounds: Dict[str, List[str]],
                 continuous_bounds: Dict[str, tuple[float, float]],
                 target_conditions: Optional[Dict] = None,
                 condition_bounds: Optional[Dict[str, tuple[float, float]]] = None) -> None:
        """
        Args:
            categorical_bounds: Maps each categorical feature name to list of allowed values.
            continuous_bounds: Maps each continuous feature name to (min, max).
            target_conditions: Target experimental conditions for candidate generation.
            condition_bounds: Maps condition feature names to (min, max) for model bounds.
        """
        self.encoder = FeatureEncoder(feature_names=self.FEATURE_NAMES)
        self.categorical_bounds = categorical_bounds
        self.continuous_bounds = continuous_bounds
        self.target_conditions = target_conditions or {}
        self.condition_bounds = condition_bounds or {
            'Temperature': (0.0, 200.0),
            'CO2_Concentration': (0.0, 100.0),
            'Humidity': (0.0, 100.0),
            'Flow_Rate': (0.0, 1000.0),
        }

        # Fit label encoders from categorical bounds so encode_candidate maps
        # strings to proper integer indices (0, 1, 2, ...)
        for cat_name, cat_values in categorical_bounds.items():
            if cat_values:
                le = LabelEncoder()
                le.fit([str(v) for v in cat_values])
                self.encoder.label_encoders[cat_name] = le

        # Unique category lists MUST come from the fitted LabelEncoder classes
        # to ensure encode/decode consistency (LabelEncoder sorts alphabetically,
        # so index 0 in le.classes_ is the same string that encodes to 0)
        self.unique_supports = list(self.encoder.label_encoders['Support'].classes_) if 'Support' in self.encoder.label_encoders else categorical_bounds.get('Support', ['SBA-15'])
        self.unique_amines1 = list(self.encoder.label_encoders['Amine_1_or_Additive_1'].classes_) if 'Amine_1_or_Additive_1' in self.encoder.label_encoders else categorical_bounds.get('Amine_1_or_Additive_1', ['No'])
        self.unique_amines2 = list(self.encoder.label_encoders['Amine_2_or_Additive_2'].classes_) if 'Amine_2_or_Additive_2' in self.encoder.label_encoders else categorical_bounds.get('Amine_2_or_Additive_2', ['No'])

        # Session categorical bounds: which materials the user selected for candidate generation
        # (may be a subset of unique_supports/amines if encoder was expanded with historical data)
        self.session_categorical_values = {
            'Support': categorical_bounds.get('Support', []),
            'Amine_1_or_Additive_1': categorical_bounds.get('Amine_1_or_Additive_1', []),
            'Amine_2_or_Additive_2': categorical_bounds.get('Amine_2_or_Additive_2', []),
        }
        self.session_categorical_indices = {}
        self._compute_session_indices()

        # Training data (tensors)
        # baseline: pre-trained on historical data, set once via set_baseline()
        self.baseline_X = torch.tensor([])
        self.baseline_Y = torch.tensor([])
        # new: experiment data accumulated during optimization
        self.new_X = torch.tensor([])
        self.new_Y = torch.tensor([])
        # train: concatenation of baseline + new (used for GP fitting)
        self.train_X = torch.tensor([])  # encoded points
        self.train_Y = torch.tensor([])  # observed values

        # Build optimisation bounds tensor
        self.bounds = self._init_bounds_tensor()

        # GP Model (initialized as None, will be created when data available)
        self.gp_model = None
        self.mll = None
        self._model_version = None

        # History tracker
        self.history = History()

    def _init_bounds_tensor(self) -> torch.Tensor:
        """Construct a [2 x d] bounds tensor from user-provided bounds."""
        # Categorical bounds: index range 0 … (n_categories - 1)
        n_supports = len(self.unique_supports)
        n_amines1 = len(self.unique_amines1)
        n_amines2 = len(self.unique_amines2)

        # Continuous bounds with safe fallback
        oc_min, oc_max = self.continuous_bounds.get('Organic_Content_pct', (0, 100))
        bet_min, bet_max = self.continuous_bounds.get('BET_Bare_Surface_Area_m2_g', (0, 1000))
        pore_min, pore_max = self.continuous_bounds.get('Average_Bare_Pore_Diameter_nm', (0, 20))

        # Condition bounds with safe fallback
        temp_min, temp_max = self.condition_bounds.get('Temperature', (0, 200))
        co2_min, co2_max = self.condition_bounds.get('CO2_Concentration', (0, 100))
        hum_min, hum_max = self.condition_bounds.get('Humidity', (0, 100))
        flow_min, flow_max = self.condition_bounds.get('Flow_Rate', (0, 1000))

        # Avoid zero‑range bounds
        if oc_max <= oc_min:
            oc_max = oc_min + 10
        if bet_max <= bet_min:
            bet_max = bet_min + 100
        if pore_max <= pore_min:
            pore_max = pore_min + 5
        if temp_max <= temp_min:
            temp_max = temp_min + 10
        if co2_max <= co2_min:
            co2_max = co2_min + 1
        if hum_max <= hum_min:
            hum_max = hum_min + 10
        if flow_max <= flow_min:
            flow_max = flow_min + 10

        lower = torch.tensor([
            0.0, 0.0, 0.0,
            oc_min, bet_min, pore_min,
            temp_min, co2_min, hum_min, flow_min
        ], dtype=torch.float32)
        upper = torch.tensor([
            max(0.0, n_supports - 1.0),
            max(0.0, n_amines1 - 1.0),
            max(0.0, n_amines2 - 1.0),
            oc_max, bet_max, pore_max,
            temp_max, co2_max, hum_max, flow_max
        ], dtype=torch.float32)
        return torch.stack([lower, upper])

    # ------------------------------------------------------------------
    # Session bounds: restrict candidate generation to user-selected materials
    # ------------------------------------------------------------------

    def _compute_session_indices(self) -> None:
        """Compute encoder indices for session-selected categorical values."""
        cat_dim_names = {0: 'Support', 1: 'Amine_1_or_Additive_1', 2: 'Amine_2_or_Additive_2'}
        self.session_categorical_indices = {}
        for dim, cat_name in cat_dim_names.items():
            cat_values = self.session_categorical_values.get(cat_name, [])
            if cat_name in self.encoder.label_encoders and cat_values:
                le = self.encoder.label_encoders[cat_name]
                indices = set()
                for v in cat_values:
                    sv = str(v)
                    if sv in le.classes_:
                        indices.add(int(le.transform([sv])[0]))
                self.session_categorical_indices[cat_name] = indices
            else:
                self.session_categorical_indices[cat_name] = set()

    def _is_within_session_bounds(self, X: torch.Tensor) -> bool:
        """Check if a single encoded point's categorical dims are within session bounds."""
        if X.dim() == 2:
            X = X.squeeze(0)
        cat_dim_names = {0: 'Support', 1: 'Amine_1_or_Additive_1', 2: 'Amine_2_or_Additive_2'}
        for dim, cat_name in cat_dim_names.items():
            allowed = self.session_categorical_indices.get(cat_name, set())
            if not allowed:
                continue
            idx = int(torch.round(X[dim]).item())
            if idx not in allowed:
                return False
        return True

    def _random_session_candidates(self, n: int) -> torch.Tensor:
        """Generate n random candidates with categorical dims restricted to session-selected values."""
        d = self.bounds.shape[1]
        candidates = torch.zeros(n, d, dtype=torch.float32)

        cat_dim_names = {0: 'Support', 1: 'Amine_1_or_Additive_1', 2: 'Amine_2_or_Additive_2'}
        for i in range(n):
            # Categorical dims: pick random index from session-allowed set
            for dim, cat_name in cat_dim_names.items():
                allowed = self.session_categorical_indices.get(cat_name, set())
                if allowed:
                    candidates[i, dim] = float(np.random.choice(list(allowed)))
                else:
                    candidates[i, dim] = 0.0
            # Continuous dims: sample uniformly within bounds
            for dim in self.CONTINUOUS_DIMS:
                lo, hi = self.bounds[0, dim].item(), self.bounds[1, dim].item()
                candidates[i, dim] = np.random.uniform(lo, hi)

        candidates = self._pin_conditions_to_target(candidates)
        return candidates

    def _has_enough_data(self, min_points: int = 0) -> bool:
        """Check if we have at least `min_points` observations."""
        if min_points == 0:
            return self.train_X.numel() > 0 and self.train_X.shape[0] >= min_points
        else:
            return self.train_X.numel() > 0
    def _round_categorical(self, X: torch.Tensor) -> torch.Tensor:
        """Round and clamp categorical dimensions to valid integer indices."""
        X_rounded = X.clone()
        for dim in self.CATEGORICAL_DIMS:
            X_rounded[..., dim] = torch.round(X_rounded[..., dim])
            X_rounded[..., dim] = torch.clamp(
                X_rounded[..., dim],
                self.bounds[0, dim],
                self.bounds[1, dim]
            )
        return X_rounded

    def _decode_config(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Convert a single encoded point (1‑D tensor) into a configuration dict.
        Assumes X is already within bounds. Condition features are pinned to
        the target_conditions values so candidates are generated at the target
        experimental conditions.
        """
        if X.dim() == 2:
            X = X.squeeze(0)

        support_idx = int(torch.clamp(torch.round(X[0]), 0, len(self.unique_supports) - 1).item())
        amine1_idx = int(torch.clamp(torch.round(X[1]), 0, len(self.unique_amines1) - 1).item())
        amine2_idx = int(torch.clamp(torch.round(X[2]), 0, len(self.unique_amines2) - 1).item())

        config = {
            'Support': self.unique_supports[support_idx],
            'Amine_1_or_Additive_1': self.unique_amines1[amine1_idx],
            'Amine_2_or_Additive_2': self.unique_amines2[amine2_idx],
            'Organic_Content_pct': float(X[3].item()),
            'BET_Bare_Surface_Area_m2_g': float(X[4].item()),
            'Average_Bare_Pore_Diameter_nm': float(X[5].item())
        }
        # Clamp continuous values to bounds
        config['Organic_Content_pct'] = max(self.bounds[0, 3].item(),
                                            min(self.bounds[1, 3].item(), config['Organic_Content_pct']))
        config['BET_Bare_Surface_Area_m2_g'] = max(self.bounds[0, 4].item(),
                                                   min(self.bounds[1, 4].item(), config['BET_Bare_Surface_Area_m2_g']))
        config['Average_Bare_Pore_Diameter_nm'] = max(self.bounds[0, 5].item(),
                                                      min(self.bounds[1, 5].item(),
                                                          config['Average_Bare_Pore_Diameter_nm']))
        # Round continuous values for readability
        config['Organic_Content_pct'] = round(config['Organic_Content_pct'], 1)
        config['BET_Bare_Surface_Area_m2_g'] = round(config['BET_Bare_Surface_Area_m2_g'], 2)
        config['Average_Bare_Pore_Diameter_nm'] = round(config['Average_Bare_Pore_Diameter_nm'], 2)

        # Pin condition features to target values (candidates are generated at target conditions)
        config['Temperature'] = float(self.target_conditions.get('temperature', 25.0))
        config['CO2_Concentration'] = float(self.target_conditions.get('co2Concentration', 0.04))
        config['Humidity'] = float(self.target_conditions.get('humidity', 0.0))
        config['Flow_Rate'] = float(self.target_conditions.get('flowRate', 100.0))
        config['Test_Method'] = self.target_conditions.get('testMethod', 'TGA')
        return config

    def _pin_conditions_to_target(self, X: torch.Tensor) -> torch.Tensor:
        """Pin condition dimensions (6-9) to the encoded target condition values."""
        X_pinned = X.clone()
        temp = float(self.target_conditions.get('temperature', 25.0))
        co2 = float(self.target_conditions.get('co2Concentration', 0.04))
        hum = float(self.target_conditions.get('humidity', 0.0))
        flow = float(self.target_conditions.get('flowRate', 100.0))
        # Clamp to bounds
        temp = max(self.bounds[0, 6].item(), min(self.bounds[1, 6].item(), temp))
        co2 = max(self.bounds[0, 7].item(), min(self.bounds[1, 7].item(), co2))
        hum = max(self.bounds[0, 8].item(), min(self.bounds[1, 8].item(), hum))
        flow = max(self.bounds[0, 9].item(), min(self.bounds[1, 9].item(), flow))
        X_pinned[..., 6] = temp
        X_pinned[..., 7] = co2
        X_pinned[..., 8] = hum
        X_pinned[..., 9] = flow
        return X_pinned

    def _rebuild_train_data(self) -> None:
        """Rebuild train_X/Y by concatenating baseline + new experiment data."""
        parts_X, parts_Y = [], []
        if self.baseline_X.numel() > 0:
            parts_X.append(self.baseline_X)
            parts_Y.append(self.baseline_Y)
        if self.new_X.numel() > 0:
            parts_X.append(self.new_X)
            parts_Y.append(self.new_Y)
        if parts_X:
            self.train_X = torch.cat(parts_X, dim=0)
            self.train_Y = torch.cat(parts_Y, dim=0)
        else:
            self.train_X = torch.tensor([])
            self.train_Y = torch.tensor([])

    def set_baseline(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Set the baseline training data (historical records) and fit the GP.

        This is called once at session initialization with all historical data.
        The baseline is preserved across subsequent add_experiment() calls.

        Args:
            X: Encoded feature tensor of shape (n, d).
            Y: Observed capacity tensor of shape (n, 1).
        """
        if X.numel() == 0:
            return
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        self.baseline_X = X.clone()
        self.baseline_Y = Y.clone()
        self._rebuild_train_data()
        self.fit_gp()

    def add_new_experiment(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Append a new experiment point (accumulated during optimization).

        Does NOT touch baseline data. Rebuilds train_X/Y and refits the GP.

        Args:
            X: Encoded feature tensor of shape (1, d).
            Y: Observed capacity tensor of shape (1, 1).
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        if self.new_X.numel() == 0:
            self.new_X = X
            self.new_Y = Y
        else:
            self.new_X = torch.cat([self.new_X, X], dim=0)
            self.new_Y = torch.cat([self.new_Y, Y], dim=0)
        self._rebuild_train_data()
        self.fit_gp()

    def _random_candidates(self, n: int) -> torch.Tensor:
        """Generate `n` random points uniformly in the bounds, with conditions pinned to target."""
        rand = torch.rand(n, self.bounds.shape[1], dtype=torch.float32)
        candidates = rand * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates = self._round_categorical(candidates)
        candidates = self._pin_conditions_to_target(candidates)
        return candidates

    def fit_gp(self) -> None:
        """
        Fit Gaussian Process model on current training data.
        Updates self.gp_model and self.mll.
        """
        if not self._has_enough_data():
            self.gp_model = None
            self.mll = None
            return

        # Work on copies to avoid mutating the source data
        train_X = self.train_X.clone()
        train_Y_arr = self.train_Y.clone()

        # Check for and clean NaN/infinite values in training data
        if torch.isnan(train_X).any() or torch.isinf(train_X).any():
            print("Warning: Cleaning NaN/Inf values from train_X before normalization")
            train_X = torch.nan_to_num(train_X, nan=0.0, posinf=None, neginf=None)
            
        if torch.isnan(train_Y_arr).any() or torch.isinf(train_Y_arr).any():
            print("Warning: Cleaning NaN/Inf values from train_Y before normalization")
            train_Y_arr = torch.nan_to_num(train_Y_arr, nan=0.0, posinf=None, neginf=None)

        # Normalise training data
        train_X_norm = normalize(train_X, self.bounds)
        
        # Check for and clean NaN/infinite values after normalization
        if torch.isnan(train_X_norm).any() or torch.isinf(train_X_norm).any():
            print("Warning: Cleaning NaN/Inf values from normalized train_X")
            train_X_norm = torch.nan_to_num(train_X_norm, nan=0.0, posinf=None, neginf=None)
        
        train_Y = train_Y_arr if train_Y_arr.dim() == 2 else train_Y_arr.unsqueeze(-1)
        
        # Check for and clean NaN/infinite values in Y
        if torch.isnan(train_Y).any() or torch.isinf(train_Y).any():
            print("Warning: Cleaning NaN/Inf values from train_Y")
            train_Y = torch.nan_to_num(train_Y, nan=0.0, posinf=None, neginf=None)

        try:
            # Create and fit GP
            self.gp_model = SingleTaskGP(
                train_X_norm,
                train_Y,
                outcome_transform=Standardize(m=1)
            )
            self.mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
            fit_gpytorch_mll(self.mll)
        except Exception as e:
            print(f"Error fitting GP model: {e}")
            print(f"train_X_norm shape: {train_X_norm.shape}, contains NaN: {torch.isnan(train_X_norm).any()}")
            print(f"train_Y shape: {train_Y.shape}, contains NaN: {torch.isnan(train_Y).any()}")
            # Fallback: set model to None to avoid crash
            self.gp_model = None
            self.mll = None

    def compute_prediction(self, X_candidate: torch.Tensor) -> Tuple[float, float]:
        """
        Compute mean and standard deviation prediction for a candidate point.

        Uses the current GP model if available and enough data are present,
        otherwise falls back to simple data statistics.

        Args:
            X_candidate: Tensor of shape (1, d) or (d,)

        Returns:
            Tuple of (mean, std) as Python floats.
        """
        # Ensure 2D shape: (1, d)
        if X_candidate.dim() == 1:
            X_candidate = X_candidate.unsqueeze(0)

        # Helper to compute fallback statistics from training data
        def fallback_stats() -> Tuple[float, float]:
            if len(self.train_Y) > 1:
                mean = self.train_Y.mean().item()
                # Ensure a minimum standard deviation to avoid zero std
                std = max(self.train_Y.std().item(), 0.1)
            else:
                mean = 0.0
                std = 1.0
            return mean, std

        # If GP model not ready, use fallback
        if self.gp_model is None or not self._has_enough_data():
            return fallback_stats()

        # Normalize candidate to [0,1] using stored bounds
        X_norm = normalize(X_candidate, self.bounds)

        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = self.gp_model.posterior(X_norm)
                mean = posterior.mean.squeeze().item()
                variance = posterior.variance.squeeze().item()

            # Sanitize outputs: ensure finite and non‑negative variance
            if not math.isfinite(mean):
                print(f"Warning: GP mean is {mean}; falling back to data stats")
                return fallback_stats()

            if not math.isfinite(variance) or variance < 0:
                print(f"Warning: GP variance is {variance}; using minimal variance")
                variance = 1e-6

            std = math.sqrt(variance)
            return mean, std

        except Exception as e:
            print(f"Error in GP prediction: {e}")
            mean, std = fallback_stats()
            print(f"Computed mean and std as {mean}, {std}")
            return mean, std


    def compute_ei(self, X_candidate: torch.Tensor) -> float:
        """
        Compute Expected Improvement for a candidate point.

        Args:
            X_candidate: Tensor of shape (1, d) or (d,)

        Returns:
            EI value
        """
        if X_candidate.dim() == 1:
            X_candidate = X_candidate.unsqueeze(0)

        if self.gp_model is None or not self._has_enough_data():
            return 0.0

        # Check for NaN/Inf in the input candidate
        if torch.isnan(X_candidate).any() or torch.isinf(X_candidate).any():
            print(f"Warning: NaN or Inf detected in X_candidate for EI computation: {X_candidate}")
            return 0.0

        mean, std = self.compute_prediction(X_candidate)
        
        # Validate the computed mean/std values
        if mean == -1 and std == -1:  # Special flag indicating error in prediction
            return 0.0
        
        # Check if train_Y has valid data
        if self.train_Y.numel() == 0:
            return 0.0
            
        try:
            best_f = self.train_Y.max().item()
            if np.isnan(best_f) or np.isinf(best_f):
                best_f = 0.0
        except:
            best_f = 0.0
            
        ei = max(0.0, mean - best_f)  # simple EI (no exploration term)

        # Ensure EI is a valid number
        if np.isnan(ei) or np.isinf(ei):
            ei = 0.0

        return ei

    def generate_candidates(self, n_candidates: int = 5) -> List[Dict[str, Any]]:
        """
        Propose the next `n_candidates` experiments.

        Uses BoTorch's qExpectedImprovement when enough data is available,
        otherwise falls back to random sampling. Candidates are filtered to
        only include materials within the session's selected search bounds.
        """
        if self.bounds is None:
            raise RuntimeError("Bounds not initialised.")

        # Over-sample to account for session-bounds filtering
        n_raw = max(self.Q_VALUES * 3, n_candidates * 5)

        # Stage 1: generate raw candidates
        if not self._has_enough_data(min_points=0) or self.gp_model is None:
            candidates_raw = self._random_candidates(n_raw)
        else:
            # Ensure training data dimension matches bounds
            if self.train_X.shape[1] != self.bounds.shape[1]:
                print(f"Warning: train_X dim {self.train_X.shape[1]} != bounds dim {self.bounds.shape[1]}. Falling back to random.")
                candidates_raw = self._random_candidates(n_raw)
            else:
                # Use the fitted GP model
                best_f = self.train_Y.max().item()
                qEI = qExpectedImprovement(model=self.gp_model, best_f=best_f)

                # Normalised bounds [0,1]^d
                norm_bounds = torch.tensor([[0.0] * self.bounds.shape[1],
                                            [1.0] * self.bounds.shape[1]], dtype=torch.float32)

                candidates_norm, _ = optimize_acqf(
                    acq_function=qEI,
                    bounds=norm_bounds,
                    q=n_raw,
                    num_restarts=20,
                    raw_samples=2048,
                    options={"maxiter": 200}
                )
                candidates_raw = unnormalize(candidates_norm, self.bounds)
                candidates_raw = self._round_categorical(candidates_raw)
                candidates_raw = self._pin_conditions_to_target(candidates_raw)

        # Stage 2: filter candidates to session bounds
        valid_candidates = []
        for candidate in candidates_raw:
            if self._is_within_session_bounds(candidate):
                valid_candidates.append(candidate)

        # If not enough valid candidates, fill with session-restricted random ones
        if len(valid_candidates) < n_candidates:
            extra = self._random_session_candidates(n_candidates - len(valid_candidates))
            valid_candidates.extend([extra[i] for i in range(extra.shape[0])])

        # Stage 3: decode, compute predictions, rank
        all_configs = []
        for candidate in valid_candidates:
            config = self._decode_config(candidate)
            mean, std = self.compute_prediction(candidate)
            ei = self.compute_ei(candidate)

            config['Predicted_CO2_Capacity_mmol_g'] = round(mean, 4)
            config['Uncertainty'] = round(std, 4)
            config['Expected_Improvement'] = ei

            all_configs.append(config)

        # Sort by predicted capacity descending and return top n_candidates
        all_configs.sort(key=lambda x: x['Predicted_CO2_Capacity_mmol_g'], reverse=True)
        return all_configs[:n_candidates]

    def add_experiment(self, result_data: Dict[str, Any]) -> None:
        """
        Incorporate a new experimental result into the optimiser.

        The method automatically computes predicted capacity, uncertainty, and
        expected improvement based on the current model (before adding new point),
        then updates the model with the new observation and refits the GP.

        Args:
            result_data: Dictionary with keys:
                - All feature names from FEATURE_NAMES (required)
                - 'actual_capacity' (required)
        """
        # Extract required fields
        if 'actual_capacity' not in result_data:
            raise ValueError("result_data must contain 'actual_capacity'")
        actual = result_data['actual_capacity']

        # Build configuration dict (features only)
        config = {name: result_data[name] for name in self.FEATURE_NAMES if name in result_data}
        if len(config) != len(self.FEATURE_NAMES):
            missing = set(self.FEATURE_NAMES) - set(config.keys())
            raise ValueError(f"Missing feature keys: {missing}")

        # Encode the configuration
        encoded = self.encoder.encode_candidate(config, feature_order=self.FEATURE_NAMES)
        if encoded is None:
            raise ValueError("FeatureEncoder failed to encode the configuration.")

        X_new = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)  # shape (1, d)

        # Check for NaN or infinite values in X_new
        if torch.isnan(X_new).any() or torch.isinf(X_new).any():
            print(f"Warning: Encountered NaN or Inf values in X_new: {X_new}")
            # Replace NaN with 0 and clamp infinite values
            X_new = torch.nan_to_num(X_new, nan=0.0, posinf=None, neginf=None)

        # --- Compute predictions using current model (before adding new point) ---
        pred_mean, pred_std = self.compute_prediction(X_new)
        
        # Check if prediction returned invalid values
        if pred_mean == -1 and pred_std == -1:
            pred_mean, pred_std = 0.0, 1.0  # fallback values
            
        ei = self.compute_ei(X_new)

        # --- Add to new experiment data (preserves baseline) ---
        y_new = torch.tensor([[actual]], dtype=torch.float32)

        # Check for NaN or infinite values in y_new
        if torch.isnan(y_new).any() or torch.isinf(y_new).any():
            print(f"Warning: Encountered NaN or Inf values in y_new: {y_new}")
            y_new = torch.nan_to_num(y_new, nan=0.0, posinf=None, neginf=None)

        self.add_new_experiment(X_new, y_new)

        # --- Record in history with computed metadata ---
        self.history.add_entry(
            config=config,
            actual_capacity=actual,
            predicted_capacity=pred_mean,
            uncertainty=pred_std,
            expected_improvement=ei
        )

    def get_history(self) -> History:
        """Return the history object."""
        return self.history

    # ------------------------------------------------------------------
    # Model persistence: save / load pre-trained GP model + encoder
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_hash(tensor: torch.Tensor) -> str:
        """Compute an MD5 hash of a tensor for change detection."""
        if tensor.numel() == 0:
            return "empty"
        return hashlib.md5(tensor.numpy().tobytes()).hexdigest()

    @staticmethod
    def find_latest_model(directory: str) -> Optional[str]:
        """Check if a base model exists under `directory/base/`."""
        base_dir = os.path.join(directory, 'base')
        if os.path.isdir(base_dir) and os.path.exists(os.path.join(base_dir, 'metadata.json')):
            return 'base'
        return None

    def save_model(self, directory: str, version: str = "v1") -> str:
        """
        Save the GP model, encoder, and metadata to a versioned directory.

        Returns the path to the saved directory.
        """
        save_dir = os.path.join(directory, version)
        os.makedirs(save_dir, exist_ok=True)

        # Save GP model state dict + training data needed to reconstruct the model
        if self.gp_model is not None and self.train_X.numel() > 0:
            train_X_norm = normalize(self.train_X.clone(), self.bounds)
            train_Y = self.train_Y.clone()
            if train_Y.dim() == 1:
                train_Y = train_Y.unsqueeze(-1)
            torch.save({
                'model_state_dict': self.gp_model.state_dict(),
                'train_X_norm': train_X_norm,
                'train_Y': train_Y,
            }, os.path.join(save_dir, 'gp_model.pt'))

        # Save encoder
        with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
            pickle.dump(self.encoder.label_encoders, f)

        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'n_baseline_points': self.baseline_X.shape[0] if self.baseline_X.numel() > 0 else 0,
            'n_total_points': self.train_X.shape[0] if self.train_X.numel() > 0 else 0,
            'baseline_X_hash': self._tensor_hash(self.baseline_X),
            'encoder_classes': {
                col: list(le.classes_) for col, le in self.encoder.label_encoders.items()
            },
            'feature_names': self.FEATURE_NAMES,
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return save_dir

    def load_model(self, directory: str, version: str = "v1",
                   baseline_X: Optional[torch.Tensor] = None) -> bool:
        """
        Load a pre-trained GP model and encoder from a versioned directory.

        Args:
            directory: Parent directory containing versioned subdirectories.
            version: Version string (e.g. "v1").
            baseline_X: Current baseline data tensor, used to verify the model
                        matches the current data via hash comparison.

        Returns True if loading succeeded, False otherwise.
        """
        load_dir = os.path.join(directory, version)
        if not os.path.exists(load_dir):
            return False

        # Load and validate metadata
        metadata_path = os.path.join(load_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Warning: No metadata.json in {load_dir}")
            return False
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # If baseline data is provided, verify it matches the saved model
        if baseline_X is not None and baseline_X.numel() > 0:
            current_hash = self._tensor_hash(baseline_X)
            saved_hash = metadata.get('baseline_X_hash', '')
            if current_hash != saved_hash:
                print(f"Info: Baseline data changed (hash {current_hash[:8]}... vs saved {saved_hash[:8]}...). Will retrain.")
                return False

        # Load encoder
        encoder_path = os.path.join(load_dir, 'encoder.pkl')
        if not os.path.exists(encoder_path):
            print(f"Warning: No encoder.pkl in {load_dir}")
            return False
        with open(encoder_path, 'rb') as f:
            loaded_encoders = pickle.load(f)

        # Verify encoder classes match metadata
        for col, expected_classes in metadata.get('encoder_classes', {}).items():
            if col in loaded_encoders:
                actual_classes = list(loaded_encoders[col].classes_)
                if actual_classes != expected_classes:
                    print(f"Warning: Encoder class mismatch for {col}. Model version is stale.")
                    return False

        # Apply loaded encoder
        self.encoder.label_encoders = loaded_encoders
        self.unique_supports = list(loaded_encoders['Support'].classes_) if 'Support' in loaded_encoders else self.unique_supports
        self.unique_amines1 = list(loaded_encoders['Amine_1_or_Additive_1'].classes_) if 'Amine_1_or_Additive_1' in loaded_encoders else self.unique_amines1
        self.unique_amines2 = list(loaded_encoders['Amine_2_or_Additive_2'].classes_) if 'Amine_2_or_Additive_2' in loaded_encoders else self.unique_amines2
        self.bounds = self._init_bounds_tensor()
        self._compute_session_indices()

        # Load GP model
        model_path = os.path.join(load_dir, 'gp_model.pt')
        if not os.path.exists(model_path):
            print(f"Warning: No gp_model.pt in {load_dir}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            train_X_norm = checkpoint.get('train_X_norm')
            train_Y = checkpoint.get('train_Y')

            if train_X_norm is not None and train_Y is not None:
                if train_Y.dim() == 1:
                    train_Y = train_Y.unsqueeze(-1)

                # Recreate GP model structure and load state dict
                self.gp_model = SingleTaskGP(
                    train_X_norm,
                    train_Y,
                    outcome_transform=Standardize(m=1)
                )
                self.gp_model.load_state_dict(checkpoint['model_state_dict'])
                self.gp_model.eval()
                self.mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)

                # Restore training data tensors (from normalized, unnormalize back)
                self.train_X = unnormalize(train_X_norm, self.bounds)
                self.train_Y = train_Y
            else:
                print("Warning: checkpoint missing train_X_norm or train_Y")
                return False

        except Exception as e:
            print(f"Error loading GP model: {e}")
            self.gp_model = None
            self.mll = None
            return False

        self._model_version = metadata.get('version', version)
        return True