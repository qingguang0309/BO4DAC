import math

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
        'Average_Bare_Pore_Diameter_nm'
    ]
    # Indices for categorical / continuous features (0‑based)
    CATEGORICAL_DIMS = [0, 1, 2]
    CONTINUOUS_DIMS = [3, 4, 5]
    Q_VALUES = 10
    def __init__(self,
                 categorical_bounds: Dict[str, List[str]],
                 continuous_bounds: Dict[str, tuple[float, float]],
                 target_conditions: Optional[Dict] = None) -> None:
        """
        Args:
            categorical_bounds: Maps each categorical feature name to list of allowed values.
            continuous_bounds: Maps each continuous feature name to (min, max).
            target_conditions: Optional target constraints (not used in base BO).
        """
        self.encoder = FeatureEncoder(feature_names=self.FEATURE_NAMES)
        self.categorical_bounds = categorical_bounds
        self.continuous_bounds = continuous_bounds
        self.target_conditions = target_conditions or {}

        # Training data (tensors)
        self.train_X = torch.tensor([])  # encoded points
        self.train_Y = torch.tensor([])  # observed values

        # Unique category lists (for decoding and bound clamping)
        self.unique_supports = categorical_bounds.get('Support', ['SBA-15'])
        self.unique_amines1 = categorical_bounds.get('Amine_1_or_Additive_1', ['No'])
        self.unique_amines2 = categorical_bounds.get('Amine_2_or_Additive_2', ['No'])

        # Build optimisation bounds tensor
        self.bounds = self._init_bounds_tensor()

        # GP Model (initialized as None, will be created when data available)
        self.gp_model = None
        self.mll = None

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

        # Avoid zero‑range bounds
        if oc_max <= oc_min:
            oc_max = oc_min + 10
        if bet_max <= bet_min:
            bet_max = bet_min + 100
        if pore_max <= pore_min:
            pore_max = pore_min + 5

        lower = torch.tensor([
            0.0, 0.0, 0.0,
            oc_min, bet_min, pore_min
        ], dtype=torch.float32)
        upper = torch.tensor([
            max(0.0, n_supports - 1.0),
            max(0.0, n_amines1 - 1.0),
            max(0.0, n_amines2 - 1.0),
            oc_max, bet_max, pore_max
        ], dtype=torch.float32)
        return torch.stack([lower, upper])

    def _has_enough_data(self, min_points: int = 3) -> bool:
        """Check if we have at least `min_points` observations."""
        return len(self.train_X) >= min_points

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
        Assumes X is already within bounds.
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
        # Clamp continuous values to bounds (already done by generation, but safe)
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
        return config

    def _random_candidates(self, n: int) -> torch.Tensor:
        """Generate `n` random points uniformly in the bounds."""
        rand = torch.rand(n, self.bounds.shape[1], dtype=torch.float32)
        candidates = rand * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return self._round_categorical(candidates)

    def fit_gp(self) -> None:
        """
        Fit Gaussian Process model on current training data.
        Updates self.gp_model and self.mll.
        """
        if not self._has_enough_data():
            self.gp_model = None
            self.mll = None
            return

        # Check for and clean NaN/infinite values in training data
        if torch.isnan(self.train_X).any() or torch.isinf(self.train_X).any():
            print("Warning: Cleaning NaN/Inf values from train_X before normalization")
            self.train_X = torch.nan_to_num(self.train_X, nan=0.0, posinf=None, neginf=None)
            
        if torch.isnan(self.train_Y).any() or torch.isinf(self.train_Y).any():
            print("Warning: Cleaning NaN/Inf values from train_Y before normalization")
            self.train_Y = torch.nan_to_num(self.train_Y, nan=0.0, posinf=None, neginf=None)

        # Normalise training data
        train_X_norm = normalize(self.train_X, self.bounds)
        
        # Check for and clean NaN/infinite values after normalization
        if torch.isnan(train_X_norm).any() or torch.isinf(train_X_norm).any():
            print("Warning: Cleaning NaN/Inf values from normalized train_X")
            train_X_norm = torch.nan_to_num(train_X_norm, nan=0.0, posinf=None, neginf=None)
        
        train_Y = self.train_Y if self.train_Y.dim() == 2 else self.train_Y.unsqueeze(-1)
        
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
        otherwise falls back to random sampling.
        """
        if self.bounds is None:
            raise RuntimeError("Bounds not initialised.")

        # Stage 1: generate raw candidates (twice as many for later filtering)
        if not self._has_enough_data() or self.gp_model is None:
            candidates_raw = self._random_candidates(self.Q_VALUES)
        else:
            # Ensure training data dimension matches bounds
            if self.train_X.shape[1] != self.bounds.shape[1]:
                raise RuntimeError(
                    f"Warning: train_X dim {self.train_X.shape[1]} != bounds dim {self.bounds.shape[1]}. Falling back to random.")
                # candidates_raw = self._random_candidates(self.Q_VALUES)
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
                    q=self.Q_VALUES,
                    num_restarts=20,
                    raw_samples=2048,
                    options={"maxiter": 200}
                )
                candidates_raw = unnormalize(candidates_norm, self.bounds)
                candidates_raw = self._round_categorical(candidates_raw)

        # Stage 2: decode, compute predictions, rank by EI
        all_configs = []
        for candidate in candidates_raw:
            config = self._decode_config(candidate)
            # Use compute_prediction method
            mean, std = self.compute_prediction(candidate)
            ei = self.compute_ei(candidate)

            config['Predicted_CO2_Capacity_mmol_g'] = round(mean, 4)

            config['Uncertainty'] = round(std, 4)
            config['Expected_Improvement'] = ei

            all_configs.append(config)

        # Sort by EI descending and return top n_candidates
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

        # --- Update training data ---
        y_new = torch.tensor([[actual]], dtype=torch.float32)
        
        # Check for NaN or infinite values in y_new
        if torch.isnan(y_new).any() or torch.isinf(y_new).any():
            print(f"Warning: Encountered NaN or Inf values in y_new: {y_new}")
            # Replace NaN with 0 and clamp infinite values
            y_new = torch.nan_to_num(y_new, nan=0.0, posinf=None, neginf=None)

        if self.train_X.numel() == 0:
            self.train_X = X_new
            self.train_Y = y_new
        else:
            self.train_X = torch.cat([self.train_X, X_new], dim=0)
            self.train_Y = torch.cat([self.train_Y, y_new], dim=0)

        # --- Refit GP with updated data ---
        self.fit_gp()

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