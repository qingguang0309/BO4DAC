import math
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
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

from encoder import FeatureEncoder

class CatalystBOWithHistory:
    """
    Bayesian Optimisation system using BoTorch.
    Categorical encoding is handled by a FeatureEncoder containing LabelEncoders.
    """

    FEATURE_NAMES = [
        'Support',
        'Amine_1_or_Additive_1',
        'Amine_2_or_Additive_2',
        'Amine_3_or_Additive_3',
        'MW_Mn_g_mol',
        'Organic_Content_pct',
        'BET_Bare_Surface_Area_m2_g',
        'Average_Bare_Pore_Diameter_nm'
    ]

    def __init__(self,
                 target_conditions: Optional[Dict] = None,
                 categorical_bounds: Optional[Dict] = None,
                 continuous_bounds: Optional[Dict] = None):
        self.target_conditions = target_conditions or {}
        self.user_categorical_bounds = categorical_bounds or {}
        self.user_continuous_bounds = continuous_bounds or {}

        # Internal encoder
        self.feature_names = self.FEATURE_NAMES
        self.encoder = FeatureEncoder()
        self.encoder.feature_names = self.FEATURE_NAMES

        # Training data placeholders
        self.train_X = torch.tensor([])
        self.train_Y = torch.tensor([])

        # Unique values for categorical features
        self.unique_supports = []
        self.unique_amines1 = []
        self.unique_amines2 = []
        self.unique_amines3 = []

        # Optimisation bounds (tensor)
        self.bounds = None
        self.categorical_dims = [0, 1, 2, 3]
        self.continuous_dims = [4, 5, 6, 7]

        # History tracking
        self.optimization_history = []

        # If bounds provided, initialise
        if self.user_categorical_bounds:
            self._initialize_bounds_and_unique_values()
            self._fit_encoder_from_bounds()

    def _has_enough_data(self):
        return self.train_X.numel() > 0 and len(self.train_X) >= 3

    def _initialize_bounds_and_unique_values(self):
        self.unique_supports = self.user_categorical_bounds.get('Support', ['SBA-15'])
        self.unique_amines1 = self.user_categorical_bounds.get('Amine_1_or_Additive_1', ['No'])
        self.unique_amines2 = self.user_categorical_bounds.get('Amine_2_or_Additive_2', ['No'])
        self.unique_amines3 = self.user_categorical_bounds.get('Amine_3_or_Additive_3', ['No'])

        n_supports = len(self.unique_supports)
        n_amines1 = len(self.unique_amines1)
        n_amines2 = len(self.unique_amines2)
        n_amines3 = len(self.unique_amines3)

        mw_min, mw_max = self.user_continuous_bounds.get('MW_Mn_g_mol', (0, 10000))
        oc_min, oc_max = self.user_continuous_bounds.get('Organic_Content_pct', (0, 100))
        bet_min, bet_max = self.user_continuous_bounds.get('BET_Bare_Surface_Area_m2_g', (0, 1000))
        pore_min, pore_max = self.user_continuous_bounds.get('Average_Bare_Pore_Diameter_nm', (0, 20))

        if mw_max <= mw_min:
            mw_max = mw_min + 1000
        if oc_max <= oc_min:
            oc_max = oc_min + 10
        if bet_max <= bet_min:
            bet_max = bet_min + 100
        if pore_max <= pore_min:
            pore_max = pore_min + 5

        self.bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, mw_min, oc_min, bet_min, pore_min],
            [max(0.0, n_supports - 1.0),
             max(0.0, n_amines1 - 1.0),
             max(0.0, n_amines2 - 1.0),
             max(0.0, n_amines3 - 1.0),
             mw_max, oc_max, bet_max, pore_max]
        ], dtype=torch.float32)

    def _fit_encoder_from_bounds(self):
        all_categories = {
            'Support': self.unique_supports,
            'Amine_1_or_Additive_1': self.unique_amines1,
            'Amine_2_or_Additive_2': self.unique_amines2,
            'Amine_3_or_Additive_3': self.unique_amines3
        }
        for col, cats in all_categories.items():
            le = LabelEncoder()
            le.fit(cats)
            self.encoder.label_encoders[col] = le
        self.encoder.feature_names = self.FEATURE_NAMES

    def generate_new_candidates(self, n_candidates: int = 5) -> List[Dict]:
        if self.bounds is None:
            raise RuntimeError("Bounds not initialised. Provide categorical_bounds and continuous_bounds.")

        if not self._has_enough_data():
            candidates_raw = self._generate_random_candidates(n_candidates * 2)
        else:
            # Check if training data dimensions match bounds dimensions
            if self.train_X.shape[1] != self.bounds.shape[1]:
                # Dimension mismatch - fall back to random candidates
                # This can happen if the training data was created with different features
                print(f"Dimension mismatch: train_X has {self.train_X.shape[1]} features, bounds expect {self.bounds.shape[1]}")
                candidates_raw = self._generate_random_candidates(n_candidates * 2)
            else:
                # Attempt BoTorch optimisation
                train_X_norm = normalize(self.train_X, self.bounds)
                train_Y = self.train_Y if self.train_Y.dim() == 2 else self.train_Y.unsqueeze(-1)

                gp = SingleTaskGP(
                    train_X_norm,
                    train_Y,
                    outcome_transform=Standardize(m=1)
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                best_f = self.train_Y.max().item()
                qEI = qExpectedImprovement(model=gp, best_f=best_f)

                # Create normalized bounds based on the actual number of features
                num_features = self.bounds.shape[1] if self.bounds is not None else 6
                norm_bounds = torch.tensor([[0.0]*num_features, [1.0]*num_features], dtype=torch.float32)
                candidates_norm, _ = optimize_acqf(
                    acq_function=qEI,
                    bounds=norm_bounds,
                    q=n_candidates * 2,
                    num_restarts=10,
                    raw_samples=1000,
                    options={"maxiter": 100}
                )
                candidates_raw = unnormalize(candidates_norm, self.bounds)
                candidates_raw = self._round_categorical(candidates_raw)

        all_configs = []
        for i in range(min(len(candidates_raw), n_candidates * 2)):
            config = self._decode_configuration(candidates_raw[i])

            # Calculate prediction even for random candidates
            pred_mean, pred_std, ei = self._predict_and_ei(candidates_raw[i:i+1])

            # Ensure we have valid prediction values even when there's insufficient data
            if not self._has_enough_data():
                # When there's not enough data, use a reasonable estimate based on existing data
                if self.train_Y.numel() > 0:
                    pred_mean = self.train_Y.mean().item()  # Use average of existing Y values
                    pred_std = max(self.train_Y.std().item(), 0.1)  # Use standard deviation of existing Y values
                    ei = 0.0  # No expected improvement without a proper model
                else:
                    pred_mean = 0.0  # Default value if no training data
                    pred_std = 1.0  # Default uncertainty
                    ei = 0.0

            config['Predicted_CO2_Capacity_mmol_g'] = float(pred_mean)
            config['Uncertainty'] = float(pred_std)
            config['Expected_Improvement'] = float(ei)
            all_configs.append(config)

        all_configs.sort(key=lambda x: x['Expected_Improvement'], reverse=True)
        return all_configs[:n_candidates]

    def _predict_and_ei(self, X_tensor):
        if not self._has_enough_data():
            return 0.0, 1.0, 0.0
        X_norm = normalize(X_tensor, self.bounds)
        train_X_norm = normalize(self.train_X, self.bounds)
        train_Y = self.train_Y if self.train_Y.dim() == 2 else self.train_Y.unsqueeze(-1)

        gp = SingleTaskGP(train_X_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = gp.posterior(X_norm)
            mean = posterior.mean.squeeze().item()
            variance = posterior.variance.squeeze().item()
            std = math.sqrt(max(variance, 1e-9))  # Ensure std is not too small or negative due to numerical issues
            best_f = self.train_Y.max().item()
            ei = max(0.0, mean - best_f)
        return mean, std, ei

    def _generate_random_candidates(self, n: int) -> torch.Tensor:
        if self.bounds is None:
            raise RuntimeError("Bounds not set; cannot generate random candidates.")
        num_features = self.bounds.shape[1]  # Get the number of features from bounds
        rand = torch.rand(n, num_features, dtype=torch.float32)
        candidates = rand * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return self._round_categorical(candidates)

    def _round_categorical(self, X: torch.Tensor) -> torch.Tensor:
        X_rounded = X.clone()
        for dim in self.categorical_dims:
            X_rounded[..., dim] = torch.round(X_rounded[..., dim])
            X_rounded[..., dim] = torch.clamp(
                X_rounded[..., dim],
                self.bounds[0, dim],
                self.bounds[1, dim]
            )
        return X_rounded

    def _decode_configuration(self, X: torch.Tensor) -> Dict:
        if X.dim() == 2:
            X = X.squeeze(0)

        support_idx = int(torch.clamp(torch.round(X[0]), 0, len(self.unique_supports)-1).item())
        amine1_idx = int(torch.clamp(torch.round(X[1]), 0, len(self.unique_amines1)-1).item())
        amine2_idx = int(torch.clamp(torch.round(X[2]), 0, len(self.unique_amines2)-1).item())
        amine3_idx = int(torch.clamp(torch.round(X[3]), 0, len(self.unique_amines3)-1).item())

        config = {
            'Support': self.unique_supports[support_idx],
            'Amine_1_or_Additive_1': self.unique_amines1[amine1_idx],
            'Amine_2_or_Additive_2': self.unique_amines2[amine2_idx],
            'Amine_3_or_Additive_3': self.unique_amines3[amine3_idx],
            'MW_Mn_g_mol': float(X[4].item()),
            'Organic_Content_pct': float(X[5].item()),
            'BET_Bare_Surface_Area_m2_g': float(X[6].item()),
            'Average_Bare_Pore_Diameter_nm': float(X[7].item())
        }
        config['MW_Mn_g_mol'] = max(self.bounds[0,4].item(),
                                    min(self.bounds[1,4].item(), config['MW_Mn_g_mol']))
        config['Organic_Content_pct'] = max(self.bounds[0,5].item(),
                                            min(self.bounds[1,5].item(), config['Organic_Content_pct']))
        config['BET_Bare_Surface_Area_m2_g'] = max(self.bounds[0,6].item(),
                                                   min(self.bounds[1,6].item(), config['BET_Bare_Surface_Area_m2_g']))
        config['Average_Bare_Pore_Diameter_nm'] = max(self.bounds[0,7].item(),
                                                      min(self.bounds[1,7].item(), config['Average_Bare_Pore_Diameter_nm']))
        config['Organic_Content_pct'] = round(config['Organic_Content_pct'], 1)
        config['BET_Bare_Surface_Area_m2_g'] = round(config['BET_Bare_Surface_Area_m2_g'], 2)
        config['Average_Bare_Pore_Diameter_nm'] = round(config['Average_Bare_Pore_Diameter_nm'], 2)
        return config

    def add_experimental_result(self, config: Dict, actual_capacity: float,
                                original_uncertainty=None,
                                original_expected_improvement=None,
                                original_predicted_capacity=None):
        encoded = self.encoder.encode_candidate(config, feature_order=self.FEATURE_NAMES)
        if encoded is not None:
            new_X = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
            new_y = torch.tensor([[actual_capacity]], dtype=torch.float32)
            if self.train_X.numel() == 0:
                self.train_X = new_X
                self.train_Y = new_y
            else:
                self.train_X = torch.cat([self.train_X, new_X], dim=0)
                self.train_Y = torch.cat([self.train_Y, new_y], dim=0)
            self.optimization_history.append({
                'iteration': len(self.optimization_history) + 1,
                'actual_capacity': actual_capacity,
                'predicted_capacity': original_predicted_capacity or actual_capacity,
                'uncertainty': original_uncertainty or 0.0,
                'current_best': self.train_Y.max().item()
            })

    def get_optimization_history(self) -> List[Dict]:
        return self.optimization_history