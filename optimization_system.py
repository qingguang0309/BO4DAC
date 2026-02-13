import math

import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class HistoricalDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()

        # Define categorical columns
        self.categorical_cols = [
            'Support',
            'Amine_1_or_Additive_1',
            'Amine_2_or_Additive_2',
            'Amine_3_or_Additive_3'
        ]

        # Define continuous columns
        self.continuous_cols = [
            'MW_Mn_g_mol',
            'Organic_Content_pct'
        ]

        # Define condition columns (for filtering)
        self.condition_cols = [
            'Relative_Humidity_pct',
            'CO2_Concentration_vol_pct',
            'Flow_Rate_mL_min',
            'Adsorption_Temperature_C',
            'CO2_Test_Method'
        ]

        # Target column
        self.target_col = 'CO2_Capacity_mmol_g'

    def preprocess_data(self, target_conditions: Optional[Dict] = None, categorical_bounds: Optional[Dict] = None):
        """Preprocess data and optionally filter by conditions and user-selected categorical bounds"""
        df = self.df.copy()

        # Filter out rows with missing values in required columns
        required_cols = (self.categorical_cols + self.continuous_cols +
                         self.condition_cols + [self.target_col])
        df = df[required_cols]
        # drop na for CO2_Capacity_mmol_g
        df = df[df['CO2_Capacity_mmol_g'].notna() & (df['CO2_Capacity_mmol_g'] != 0)]
        # replace categorical data 0 to be "No"
        df = df.replace(to_replace='0', value='No')

        # Filter by user-selected categorical bounds first
        if categorical_bounds:
            for col, allowed_values in categorical_bounds.items():
                if col in df.columns and allowed_values:
                    df[col] = df[col].str.replace('0','No')
                    df = df[df[col].isin(allowed_values)]
                    print(f"Filtered {col} to {len(allowed_values)} allowed values: {allowed_values}")

        # Then filter by conditions if provided
        if target_conditions:
            for col, value in target_conditions.items():
                if col in df.columns:
                    if col == 'CO2_Test_Method':
                        df = df[df[col] == value]
                    elif col in ['Relative_Humidity_pct', 'CO2_Concentration_vol_pct',
                                 'Flow_Rate_mL_min', 'Adsorption_Temperature_C']:
                        tolerance = abs(value) * 0.05
                        df = df[(df[col] >= value - tolerance) & (df[col] <= value + tolerance)]

        print(f"Using {len(df)} samples for training")

        # Encode categorical variables
        encoded_features = []
        for col in self.categorical_cols:
            le = LabelEncoder()
            encoded_col = le.fit_transform(df[col].astype(str))
            encoded_features.append(encoded_col.reshape(-1, 1))
            self.label_encoders[col] = le

        # Get continuous variables
        continuous_features = df[self.continuous_cols].values

        # Combine features
        if encoded_features:
            categorical_features = np.hstack(encoded_features)
            if len(continuous_features) > 0:
                X = np.hstack([categorical_features, continuous_features])
            else:
                X = categorical_features
        else:
            X = continuous_features

        # Target
        y = df[self.target_col].values.reshape(-1, 1)

        # Scale features if we have data
        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        print(f"Feature dimension: {X_scaled.shape[1] if len(X_scaled) > 0 else 0}")

        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), df

    def encode_configuration(self, config: Dict) -> np.ndarray:
        """Encode a configuration for prediction"""
        encoded_features = []

        for col in self.categorical_cols:
            if col in config and col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    encoded_value = le.transform([str(config[col])])[0]
                    encoded_features.append(encoded_value)
                except ValueError:
                    all_categories = set(le.classes_)
                    new_category = str(config[col])
                    if new_category not in all_categories:
                        all_categories.add(new_category)
                        expanded_categories = list(all_categories)

                        new_le = LabelEncoder()
                        new_le.fit(expanded_categories)

                        self.label_encoders[col] = new_le
                        encoded_value = new_le.transform([str(config[col])])[0]
                        encoded_features.append(encoded_value)
                    else:
                        encoded_features.append(0)
            else:
                encoded_features.append(0)

        for col in self.continuous_cols:
            if col in config:
                encoded_features.append(float(config[col]))
            else:
                encoded_features.append(0.0)

        return np.array(encoded_features).reshape(1, -1)


class CatalystBOWithHistory:
    def __init__(self,
                 data_processor: HistoricalDataProcessor,
                 target_conditions: Optional[Dict] = None,
                 categorical_bounds: Optional[Dict] = None,
                 continuous_bounds: Optional[Dict] = None):
        """
        Initialize BO system with historical data
        Args:
            data_processor: Historical data processor
            target_conditions: Experimental conditions to filter data
            categorical_bounds: User-specified bounds for categorical variables
            continuous_bounds: User-specified bounds for continuous variables
        """
        self.data_processor = data_processor
        self.target_conditions = target_conditions or {}

        self.user_categorical_bounds = categorical_bounds or {}
        self.user_continuous_bounds = continuous_bounds or {}

        self.optimization_history = []

        self._initialize_with_historical_data()

    def _initialize_with_historical_data(self):
        """Initialize with historical data for initial GP fitting"""
        self.train_X, self.train_Y, self.historical_df = self.data_processor.preprocess_data(
            self.target_conditions,
            self.user_categorical_bounds
        )

        print(f"Loaded {len(self.train_X)} historical experiments")
        if len(self.train_X) > 0:
            print(f"Best historical capacity: {self.train_Y.max().item():.4f} mmol/g")

        self._initialize_bounds_and_unique_values()

        self.real_experiments_df = pd.DataFrame(columns=self.historical_df.columns)
        self.use_real_data_only = False

    def _initialize_bounds_and_unique_values(self):
        """Initialize unique values and bounds for optimization"""
        if len(self.historical_df) == 0:
            historical_supports = []
            historical_amines1 = []
            historical_amines2 = []
            historical_amines3 = []
        else:
            historical_supports = self.historical_df['Support'].astype(str).unique().tolist()
            historical_amines1 = self.historical_df['Amine_1_or_Additive_1'].astype(str).unique().tolist()
            historical_amines2 = self.historical_df['Amine_2_or_Additive_2'].astype(str).unique().tolist()
            historical_amines3 = self.historical_df['Amine_3_or_Additive_3'].astype(str).unique().tolist()

        self.unique_supports = self.user_categorical_bounds.get('Support', historical_supports)
        self.unique_amines1 = self.user_categorical_bounds.get('Amine_1_or_Additive_1', historical_amines1)
        self.unique_amines2 = self.user_categorical_bounds.get('Amine_2_or_Additive_2', historical_amines2)
        self.unique_amines3 = self.user_categorical_bounds.get('Amine_3_or_Additive_3', historical_amines3)

        if 'Amine 1 or Additive 1' in self.user_categorical_bounds:
            self.unique_amines1 = self.user_categorical_bounds.get('Amine 1 or Additive 1', historical_amines1)
        if 'Amine 2 or Additive 2' in self.user_categorical_bounds:
            self.unique_amines2 = self.user_categorical_bounds.get('Amine 2 or Additive 2', historical_amines2)
        if 'Additive 3' in self.user_categorical_bounds:
            self.unique_amines3 = self.user_categorical_bounds.get('Additive 3', historical_amines3)

        # If no unique values exist (no data and no user bounds), provide defaults
        if len(self.unique_supports) == 0:
            self.unique_supports = ['SBA-15']
        if len(self.unique_amines1) == 0:
            self.unique_amines1 = ['No']
        if len(self.unique_amines2) == 0:
            self.unique_amines2 = ['No']
        if len(self.unique_amines3) == 0:
            self.unique_amines3 = ['No']

        print(f"Unique Supports ({len(self.unique_supports)}): {self.unique_supports[:5]}...")
        print(f"Unique Amines 1 ({len(self.unique_amines1)}): {self.unique_amines1[:5]}...")
        print(f"Unique Amines 2 ({len(self.unique_amines2)}): {self.unique_amines2[:5]}...")
        print(f"Unique Amines 3 ({len(self.unique_amines3)}): {self.unique_amines3[:5]}...")

        n_supports = len(self.unique_supports)
        n_amines1 = len(self.unique_amines1)
        n_amines2 = len(self.unique_amines2)
        n_amines3 = len(self.unique_amines3)

        # Continuous bounds
        if self.user_continuous_bounds:
            mw_min = self.user_continuous_bounds.get('MW_Mn_g_mol', [0, 10000])[0]
            mw_max = self.user_continuous_bounds.get('MW_Mn_g_mol', [0, 10000])[1]
            oc_min = self.user_continuous_bounds.get('Organic_Content_pct', [0, 100])[0]
            oc_max = self.user_continuous_bounds.get('Organic_Content_pct', [0, 100])[1]
        else:
            if len(self.historical_df) > 0:
                mw_min, mw_max = self.historical_df['MW_Mn_g_mol'].min(), self.historical_df['MW_Mn_g_mol'].max()
                oc_min, oc_max = self.historical_df['Organic_Content_pct'].min(), self.historical_df['Organic_Content_pct'].max()
                mw_padding = (mw_max - mw_min) * 0.1 if mw_max > mw_min else abs(mw_min) * 0.1
                oc_padding = (oc_max - oc_min) * 0.1 if oc_max > oc_min else abs(oc_min) * 0.1
                mw_min = mw_min - mw_padding
                mw_max = mw_max + mw_padding
                oc_min = oc_min - oc_padding
                oc_max = oc_max + oc_padding
            else:
                mw_min, mw_max = 0, 10000
                oc_min, oc_max = 0, 100

        mw_min = max(0, mw_min)
        oc_min = max(0, oc_min)

        if mw_max <= mw_min:
            mw_max = mw_min + 1000
        if oc_max <= oc_min:
            oc_max = oc_min + 10

        self.bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, mw_min, oc_min],
            [max(0, n_supports - 1.0), max(0, n_amines1 - 1.0),
             max(0, n_amines2 - 1.0), max(0, n_amines3 - 1.0),
             mw_max, oc_max]
        ], dtype=torch.float32)

        self.categorical_dims = [0, 1, 2, 3]
        self.continuous_dims = [4, 5]

        print(f"\nOptimization bounds:")
        print(f"Supports: [{self.bounds[0, 0]:.1f}, {self.bounds[1, 0]:.1f}] ({n_supports} unique)")
        print(f"Amine 1: [{self.bounds[0, 1]:.1f}, {self.bounds[1, 1]:.1f}] ({n_amines1} unique)")
        print(f"Amine 2: [{self.bounds[0, 2]:.1f}, {self.bounds[1, 2]:.1f}] ({n_amines2} unique)")
        print(f"Amine 3: [{self.bounds[0, 3]:.1f}, {self.bounds[1, 3]:.1f}] ({n_amines3} unique)")
        print(f"MW_Mn: [{self.bounds[0, 4]:.1f}, {self.bounds[1, 4]:.1f}] g/mol")
        print(f"Organic Content: [{self.bounds[0, 5]:.1f}, {self.bounds[1, 5]:.1f}]%")

        self._fit_label_encoders_for_all_categories()

    def _fit_label_encoders_for_all_categories(self):
        """Pre-fit label encoders with all possible categories to avoid unseen labels issue"""
        all_categories = {
            'Support': self.unique_supports,
            'Amine_1_or_Additive_1': self.unique_amines1,
            'Amine_2_or_Additive_2': self.unique_amines2,
            'Amine_3_or_Additive_3': self.unique_amines3
        }

        for col, categories in all_categories.items():
            le = LabelEncoder()
            le.fit(categories)
            self.data_processor.label_encoders[col] = le

    def _update_training_data(self):
        """Update training data based on whether to use only real data"""
        if self.use_real_data_only:
            filtered_historical = self._filter_historical_data_by_conditions()
            combined_df = pd.concat([filtered_historical, self.real_experiments_df], ignore_index=True)
            self._preprocess_combined_data(combined_df)
        else:
            combined_df = pd.concat([self.historical_df, self.real_experiments_df], ignore_index=True)
            self._preprocess_combined_data(combined_df)

    def _preprocess_real_data(self):
        """Preprocess only real experimental data (no minimum sample requirement)."""
        if len(self.real_experiments_df) == 0:
            self.train_X = torch.tensor([])
            self.train_Y = torch.tensor([])
            return

        df = self.real_experiments_df.copy()

        filtered_df = df.copy()
        for col, allowed_values in self.user_categorical_bounds.items():
            if col in filtered_df.columns and allowed_values:
                filtered_df = filtered_df[filtered_df[col].isin(allowed_values)]
                print(f"Filtered real data {col} to {len(allowed_values)} allowed values: {allowed_values}")

        encoded_features = []
        for col in self.data_processor.categorical_cols:
            le = self.data_processor.label_encoders.get(col)
            if le is None:
                le = LabelEncoder()
                le.fit(filtered_df[col].astype(str))
                self.data_processor.label_encoders[col] = le

            try:
                encoded_col = le.transform(filtered_df[col].astype(str))
            except ValueError:
                all_possible_labels = set(le.classes_) if hasattr(le, 'classes_') else set()
                new_labels = set(filtered_df[col].astype(str).unique())
                all_labels = all_possible_labels.union(new_labels)
                new_le = LabelEncoder()
                new_le.fit(list(all_labels))
                self.data_processor.label_encoders[col] = new_le
                encoded_col = new_le.transform(filtered_df[col].astype(str))

            encoded_features.append(encoded_col.reshape(-1, 1))

        continuous_features = filtered_df[self.data_processor.continuous_cols].values

        if encoded_features:
            categorical_features = np.hstack(encoded_features)
            X = np.hstack([categorical_features, continuous_features])
        else:
            X = continuous_features

        y = filtered_df[self.data_processor.target_col].values.reshape(-1, 1)

        if len(X) == 0:
            self.train_X = torch.tensor([])
            self.train_Y = torch.tensor([])
            return

        X_scaled = self.data_processor.scaler.fit_transform(X)

        self.train_X = torch.tensor(X_scaled, dtype=torch.float32)
        self.train_Y = torch.tensor(y, dtype=torch.float32)

    def _preprocess_combined_data(self, df):
        """Preprocess combined historical and real data (no minimum sample requirement)."""
        if len(df) == 0:
            self.train_X = torch.tensor([])
            self.train_Y = torch.tensor([])
            return

        filtered_df = df.copy()
        for col, allowed_values in self.user_categorical_bounds.items():
            if col in filtered_df.columns and allowed_values:
                filtered_df = filtered_df[filtered_df[col].isin(allowed_values)]
                print(f"Filtered combined data {col} to {len(allowed_values)} allowed values: {allowed_values}")

        encoded_features = []
        for col in self.data_processor.categorical_cols:
            le = self.data_processor.label_encoders.get(col)
            if le is None:
                le = LabelEncoder()
                le.fit(filtered_df[col].astype(str))
                self.data_processor.label_encoders[col] = le

            encoded_col = le.transform(filtered_df[col].astype(str))
            encoded_features.append(encoded_col.reshape(-1, 1))

        continuous_features = filtered_df[self.data_processor.continuous_cols].values
        categorical_features = np.hstack(encoded_features)
        X = np.hstack([categorical_features, continuous_features])

        y = filtered_df[self.data_processor.target_col].values.reshape(-1, 1)

        if len(X) == 0:
            self.train_X = torch.tensor([])
            self.train_Y = torch.tensor([])
            return

        X_scaled = self.data_processor.scaler.fit_transform(X)

        self.train_X = torch.tensor(X_scaled, dtype=torch.float32)
        self.train_Y = torch.tensor(y, dtype=torch.float32)

    def _filter_historical_data_by_conditions(self):
        """Filter historical data based on current search space and conditions"""
        if len(self.historical_df) == 0:
            return self.historical_df.copy()

        filtered_df = self.historical_df.copy()

        for col, allowed_values in self.user_categorical_bounds.items():
            if col in filtered_df.columns and allowed_values:
                filtered_df = filtered_df[filtered_df[col].isin(allowed_values)]
                print(f"Filtered historical data {col} to {len(allowed_values)} allowed values: {allowed_values}")

        if not filtered_df.empty:
            target_temp = self.target_conditions.get('Adsorption_Temperature_C', 25)
            target_rh = self.target_conditions.get('Relative_Humidity_pct', 0)
            target_co2_conc = self.target_conditions.get('CO2_Concentration_vol_pct', 0.04)

            temp_tolerance = 10
            rh_tolerance = 20
            co2_tolerance = 0.02

            temp_mask = abs(filtered_df['Adsorption_Temperature_C'] - target_temp) <= temp_tolerance
            rh_mask = abs(filtered_df['Relative_Humidity_pct'] - target_rh) <= rh_tolerance
            co2_mask = abs(filtered_df['CO2_Concentration_vol_pct'] - target_co2_conc) <= co2_tolerance

            combined_mask = temp_mask | rh_mask | co2_mask

            if combined_mask.sum() < 5:
                temp_tolerance *= 2
                rh_tolerance *= 2
                co2_tolerance *= 2
                temp_mask = abs(filtered_df['Adsorption_Temperature_C'] - target_temp) <= temp_tolerance
                rh_mask = abs(filtered_df['Relative_Humidity_pct'] - target_rh) <= rh_tolerance
                co2_mask = abs(filtered_df['CO2_Concentration_vol_pct'] - target_co2_conc) <= co2_tolerance
                combined_mask = temp_mask | rh_mask | co2_mask

            if combined_mask.sum() < 5:
                return filtered_df
            else:
                return filtered_df[combined_mask].copy()
        else:
            return filtered_df

    def encode_configuration_for_bo(self, config_dict: Dict) -> torch.Tensor:
        """Encode a configuration to tensor for BO optimization"""
        X = torch.zeros(6)

        try:
            X[0] = float(self.unique_supports.index(str(config_dict.get('Support', ''))))
        except ValueError:
            X[0] = 0.0
        try:
            X[1] = float(self.unique_amines1.index(str(config_dict.get('Amine_1_or_Additive_1', ''))))
        except ValueError:
            X[1] = 0.0
        try:
            X[2] = float(self.unique_amines2.index(str(config_dict.get('Amine_2_or_Additive_2', ''))))
        except ValueError:
            X[2] = 0.0
        try:
            X[3] = float(self.unique_amines3.index(str(config_dict.get('Amine_3_or_Additive_3', ''))))
        except ValueError:
            X[3] = 0.0

        X[4] = float(config_dict.get('MW_Mn_g_mol', 0.0))
        X[5] = float(config_dict.get('Organic_Content_pct', 0.0))

        return X.unsqueeze(0)

    def decode_configuration(self, X: torch.Tensor) -> Dict:
        """Decode tensor to configuration dictionary"""
        if X.dim() == 2:
            X = X.squeeze(0)

        support_idx = max(0, min(len(self.unique_supports) - 1, int(X[0].round().item())))
        amine1_idx = max(0, min(len(self.unique_amines1) - 1, int(X[1].round().item())))
        amine2_idx = max(0, min(len(self.unique_amines2) - 1, int(X[2].round().item())))
        amine3_idx = max(0, min(len(self.unique_amines3) - 1, int(X[3].round().item())))

        config = {
            'Support': self.unique_supports[support_idx],
            'Amine_1_or_Additive_1': self.unique_amines1[amine1_idx],
            'Amine_2_or_Additive_2': self.unique_amines2[amine2_idx],
            'Amine_3_or_Additive_3': self.unique_amines3[amine3_idx],
            'MW_Mn_g_mol': float(X[4].item()),
            'Organic_Content_pct': float(X[5].item())
        }

        config['MW_Mn_g_mol'] = max(self.bounds[0, 4].item(),
                                    min(self.bounds[1, 4].item(), config['MW_Mn_g_mol']))
        config['Organic_Content_pct'] = max(self.bounds[0, 5].item(),
                                            min(self.bounds[1, 5].item(), config['Organic_Content_pct']))
        config['Organic_Content_pct'] = round(config['Organic_Content_pct'], 1)
        config.update(self.target_conditions)

        return config

    def round_categorical(self, X: torch.Tensor) -> torch.Tensor:
        """Round categorical variables to nearest integer index"""
        X_rounded = X.clone()
        for dim in self.categorical_dims:
            X_rounded[..., dim] = torch.round(X_rounded[..., dim])
            X_rounded[..., dim] = torch.clamp(
                X_rounded[..., dim],
                self.bounds[0, dim],
                self.bounds[1, dim]
            )
        return X_rounded

    def transform_for_gp(self, X: torch.Tensor) -> torch.Tensor:
        """Transform BO configuration to GP input space"""
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X

        configs = []
        for i in range(X_np.shape[0]):
            if X_np.ndim == 1:
                x = X_np
            else:
                x = X_np[i]

            support_idx = max(0, min(len(self.unique_supports) - 1, int(x[0])))
            amine1_idx = max(0, min(len(self.unique_amines1) - 1, int(x[1])))
            amine2_idx = max(0, min(len(self.unique_amines2) - 1, int(x[2])))
            amine3_idx = max(0, min(len(self.unique_amines3) - 1, int(x[3])))

            config = {
                'Support': self.unique_supports[support_idx],
                'Amine_1_or_Additive_1': self.unique_amines1[amine1_idx],
                'Amine_2_or_Additive_2': self.unique_amines2[amine2_idx],
                'Amine_3_or_Additive_3': self.unique_amines3[amine3_idx],
                'MW_Mn_g_mol': float(x[4]),
                'Organic_Content_pct': float(x[5])
            }
            configs.append(config)

        encoded = []
        for config in configs:
            try:
                encoded_config = self.data_processor.encode_configuration(config)
                encoded.append(encoded_config)
            except ValueError as e:
                print(f"Warning: Error encoding configuration {config}: {e}")
                encoded.append(np.zeros((1, len(self.data_processor.categorical_cols) + len(self.data_processor.continuous_cols))))

        X_gp = np.vstack(encoded)

        if not hasattr(self.data_processor.scaler, 'mean_'):
            X_gp_scaled = self.data_processor.scaler.fit_transform(X_gp)
        else:
            X_gp_scaled = self.data_processor.scaler.transform(X_gp)

        return torch.tensor(X_gp_scaled, dtype=torch.float32)

    def generate_new_candidates(self, n_candidates: int = 1) -> List[Dict]:
        """Generate new catalyst candidates using Bayesian Optimization or random sampling if no data."""
        self._update_training_data()

        print(f"\nCurrent training data: {len(self.train_X)} points")
        if len(self.train_X) > 0:
            print(f"Current best capacity: {self.train_Y.max().item():.4f} mmol/g")

        candidates_to_generate = max(n_candidates * 2, 10)
        gp = None

        if not self._are_bounds_valid() or len(self.train_X) < 3:
            print("Warning: Not enough training data or invalid bounds. Generating random candidates.")
            candidates = self._generate_random_candidates(candidates_to_generate)
        else:
            try:
                train_X_normalized = normalize(self.train_X, self.bounds)
                train_Y_reshaped = self.train_Y if self.train_Y.dim() == 2 else self.train_Y.unsqueeze(-1)
                gp = SingleTaskGP(
                    train_X_normalized,
                    train_Y_reshaped,
                    outcome_transform=Standardize(m=1)
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                best_f = self.train_Y.max().item()
                qEI = qExpectedImprovement(model=gp, best_f=best_f)

                norm_bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.float32)
                candidates_norm, acq_values = optimize_acqf(
                    acq_function=qEI,
                    bounds=norm_bounds,
                    q=candidates_to_generate,
                    num_restarts=10,
                    raw_samples=1000,
                    options={"maxiter": 100}
                )

                candidates = unnormalize(candidates_norm, self.bounds)
                candidates = self.round_categorical(candidates)

            except Exception as e:
                print(f"Warning: Optimization failed: {e}")
                candidates = self._generate_random_candidates(candidates_to_generate)

        if gp is None and len(self.train_X) >= 3:
            try:
                train_X_normalized = normalize(self.train_X, self.bounds)
                train_Y_reshaped = self.train_Y if self.train_Y.dim() == 2 else self.train_Y.unsqueeze(-1)
                gp = SingleTaskGP(train_X_normalized, train_Y_reshaped)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
            except:
                pass

        best_f_val = self.train_Y.max().item() if len(self.train_X) > 0 else 0.0
        all_candidate_configs = []

        for i in range(min(candidates_to_generate, len(candidates))):
            try:
                config = self.decode_configuration(candidates[i])

                pred_mean = 0.0
                pred_std = 0.0
                ei = 0.0
                if gp is not None:
                    candidate_normalized = normalize(candidates[i:i+1], self.bounds)
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        posterior = gp.posterior(candidate_normalized)
                        pred_mean = posterior.mean.squeeze().item()
                        pred_std = math.sqrt(posterior.variance.squeeze().item())
                        ei = max(0, pred_mean - best_f_val)

                config['Organic_Content_pct'] = round(config['Organic_Content_pct'], 1)
                config['Predicted_CO2_Capacity_mmol_g'] = float(pred_mean)
                config['Uncertainty'] = float(pred_std)
                config['Expected_Improvement'] = float(ei)
                config['Is_New'] = not self.check_similar_configuration(config)

                all_candidate_configs.append(config)

            except Exception as e:
                print(f"Warning: Failed to process candidate {i}: {e}")
                continue

        if all_candidate_configs:
            sorted_candidates = sorted(
                all_candidate_configs,
                key=lambda x: x['Expected_Improvement'],
                reverse=True
            )
            top_candidates = sorted_candidates[:n_candidates]
        else:
            top_candidates = []

        print(f"\nGenerated {len(top_candidates)} candidates")
        if top_candidates:
            print(f"Best predicted capacity: {top_candidates[0]['Predicted_CO2_Capacity_mmol_g']:.4f} mmol/g")

        return top_candidates

    def _are_bounds_valid(self) -> bool:
        """Check if bounds are valid for optimization"""
        for i in range(self.bounds.shape[1]):
            if self.bounds[1, i] <= self.bounds[0, i]:
                return False
        return True

    def _generate_random_candidates(self, n_candidates: int) -> torch.Tensor:
        """Generate random candidates as fallback."""
        candidates = torch.rand(n_candidates, 6, dtype=torch.float32)
        candidates = candidates * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates = self.round_categorical(candidates)
        return candidates

    def check_similar_configuration(self, config: Dict, tolerance: float = 1e-2) -> bool:
        """Check if similar configuration exists in historical or real data"""
        for idx, row in self.historical_df.iterrows():
            if self._is_similar(row, config, tolerance):
                return True
        for idx, row in self.real_experiments_df.iterrows():
            if self._is_similar(row, config, tolerance):
                return True
        return False

    def _is_similar(self, row, config: Dict, tolerance: float) -> bool:
        return (str(row['Support']) == config['Support'] and
                str(row['Amine_1_or_Additive_1']) == config['Amine_1_or_Additive_1'] and
                str(row['Amine_2_or_Additive_2']) == config['Amine_2_or_Additive_2'] and
                str(row['Amine_3_or_Additive_3']) == config['Amine_3_or_Additive_3'] and
                abs(row['MW_Mn_g_mol'] - config['MW_Mn_g_mol']) < tolerance and
                abs(row['Organic_Content_pct'] - config['Organic_Content_pct']) < tolerance)

    def add_experimental_result(self, config: Dict, actual_capacity: float, original_uncertainty=None,
                                original_expected_improvement=None, original_predicted_capacity=None):
        """Add new experimental result to real data"""
        new_row = config.copy()
        new_row['CO2_Capacity_mmol_g'] = actual_capacity
        new_row['Relative_Humidity_pct'] = self.target_conditions.get('Relative_Humidity_pct', 0)
        new_row['CO2_Concentration_vol_pct'] = self.target_conditions.get('CO2_Concentration_vol_pct', 0)
        new_row['Flow_Rate_mL_min'] = self.target_conditions.get('Flow_Rate_mL_min', 0)
        new_row['Adsorption_Temperature_C'] = self.target_conditions.get('Adsorption_Temperature_C', 0)
        new_row['CO2_Test_Method'] = self.target_conditions.get('CO2_Test_Method', 'TGA')

        self.real_experiments_df = pd.concat([self.real_experiments_df, pd.DataFrame([new_row])], ignore_index=True)

        print(f"\nAdded new real experiment: {actual_capacity:.4f} mmol/g")
        print(f"Total real experiments: {len(self.real_experiments_df)}")

        self._track_optimization_progress(actual_capacity, original_uncertainty,
                                          original_expected_improvement, original_predicted_capacity)

        self.archive_old_experiments(keep_last_n=50)
        self.archive_old_history(keep_last_n=100)

    def _track_optimization_progress(self, actual_capacity: float, original_uncertainty=None,
                                     original_expected_improvement=None, original_predicted_capacity=None):
        """Track optimization progress for visualization"""
        iteration = len(self.real_experiments_df)   # iteration = number of real experiments

        all_capacities = []
        if len(self.train_Y) > 0:
            all_capacities.extend(self.train_Y.flatten().tolist())
        if len(self.real_experiments_df) > 0:
            all_capacities.extend(self.real_experiments_df['CO2_Capacity_mmol_g'].tolist())

        current_best = max(all_capacities) if all_capacities else 0.0

        if original_uncertainty is not None:
            uncertainty = original_uncertainty
        else:
            uncertainty = 0.0

        if original_predicted_capacity is not None:
            predicted_capacity_for_this_experiment = original_predicted_capacity
        else:
            predicted_capacity_for_this_experiment = actual_capacity

        history_entry = {
            'iteration': iteration,
            'actual_capacity': actual_capacity,
            'predicted_capacity': predicted_capacity_for_this_experiment,
            'current_best': current_best,
            'uncertainty': uncertainty,
            'total_experiments': len(self.real_experiments_df),
            'total_data_points': len(self.train_X)
        }

        self.optimization_history.append(history_entry)

        MAX_HISTORY_SIZE = 100
        if len(self.optimization_history) > MAX_HISTORY_SIZE:
            self.optimization_history = self.optimization_history[-MAX_HISTORY_SIZE:]

        print(f"Optimization history updated: iteration={iteration}, best={current_best:.4f}, "
              f"predicted={predicted_capacity_for_this_experiment:.4f}, uncertainty={uncertainty:.4f}")

    def get_optimization_history(self) -> List[Dict]:
        return self.optimization_history

    def archive_old_experiments(self, keep_last_n: int = 50):
        if len(self.real_experiments_df) > keep_last_n:
            rows_to_keep = self.real_experiments_df.tail(keep_last_n)
            archived_count = len(self.real_experiments_df) - len(rows_to_keep)
            self.real_experiments_df = rows_to_keep.reset_index(drop=True)
            print(f"Archived {archived_count} old experiments, keeping {len(self.real_experiments_df)} most recent")

    def archive_old_history(self, keep_last_n: int = 100):
        if len(self.optimization_history) > keep_last_n:
            entries_to_keep = self.optimization_history[-keep_last_n:]
            archived_count = len(self.optimization_history) - len(entries_to_keep)
            self.optimization_history = entries_to_keep
            print(f"Archived {archived_count} old history entries, keeping {len(self.optimization_history)} most recent")

    def update_bounds_and_conditions(self, new_target_conditions: Dict, new_categorical_bounds: Dict,
                                     new_continuous_bounds: Dict):
        """Update the target conditions and bounds for the optimizer"""
        self.target_conditions = new_target_conditions
        self.user_categorical_bounds = new_categorical_bounds
        self.user_continuous_bounds = new_continuous_bounds

        if len(self.historical_df) > 0:
            historical_supports = self.historical_df['Support'].unique().tolist()
            historical_amines1 = self.historical_df['Amine_1_or_Additive_1'].unique().tolist()
            historical_amines2 = self.historical_df['Amine_2_or_Additive_2'].unique().tolist()
            historical_amines3 = self.historical_df['Amine_3_or_Additive_3'].unique().tolist()
        else:
            historical_supports = []
            historical_amines1 = []
            historical_amines2 = []
            historical_amines3 = []

        self.unique_supports = self.user_categorical_bounds.get('Support', historical_supports)
        self.unique_amines1 = self.user_categorical_bounds.get('Amine_1_or_Additive_1', historical_amines1)
        self.unique_amines2 = self.user_categorical_bounds.get('Amine_2_or_Additive_2', historical_amines2)
        self.unique_amines3 = self.user_categorical_bounds.get('Amine_3_or_Additive_3', historical_amines3)

        if 'Amine 1 or Additive 1' in self.user_categorical_bounds:
            self.unique_amines1 = self.user_categorical_bounds.get('Amine 1 or Additive 1', historical_amines1)
        if 'Amine 2 or Additive 2' in self.user_categorical_bounds:
            self.unique_amines2 = self.user_categorical_bounds.get('Amine 2 or Additive 2', historical_amines2)
        if 'Additive 3' in self.user_categorical_bounds:
            self.unique_amines3 = self.user_categorical_bounds.get('Additive 3', historical_amines3)

        if len(self.unique_supports) == 0:
            self.unique_supports = ['SBA-15']
        if len(self.unique_amines1) == 0:
            self.unique_amines1 = ['No']
        if len(self.unique_amines2) == 0:
            self.unique_amines2 = ['No']
        if len(self.unique_amines3) == 0:
            self.unique_amines3 = ['No']

        n_supports = len(self.unique_supports)
        n_amines1 = len(self.unique_amines1)
        n_amines2 = len(self.unique_amines2)
        n_amines3 = len(self.unique_amines3)

        if self.user_continuous_bounds:
            mw_min = self.user_continuous_bounds.get('MW_Mn_g_mol', [0, 10000])[0]
            mw_max = self.user_continuous_bounds.get('MW_Mn_g_mol', [0, 10000])[1]
            oc_min = self.user_continuous_bounds.get('Organic_Content_pct', [0, 100])[0]
            oc_max = self.user_continuous_bounds.get('Organic_Content_pct', [0, 100])[1]
        else:
            if len(self.historical_df) > 0:
                mw_min, mw_max = self.historical_df['MW_Mn_g_mol'].min(), self.historical_df['MW_Mn_g_mol'].max()
                oc_min, oc_max = self.historical_df['Organic_Content_pct'].min(), self.historical_df['Organic_Content_pct'].max()
                mw_padding = (mw_max - mw_min) * 0.1 if mw_max > mw_min else abs(mw_min) * 0.1
                oc_padding = (oc_max - oc_min) * 0.1 if oc_max > oc_min else abs(oc_min) * 0.1
                mw_min = mw_min - mw_padding
                mw_max = mw_max + mw_padding
                oc_min = oc_min - oc_padding
                oc_max = oc_max + oc_padding
            else:
                mw_min, mw_max = 0, 10000
                oc_min, oc_max = 0, 100

        mw_min = max(0, mw_min)
        oc_min = max(0, oc_min)
        if mw_max <= mw_min:
            mw_max = mw_min + 1000
        if oc_max <= oc_min:
            oc_max = oc_min + 10

        self.bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, mw_min, oc_min],
            [max(0, n_supports - 1.0), max(0, n_amines1 - 1.0),
             max(0, n_amines2 - 1.0), max(0, n_amines3 - 1.0),
             mw_max, oc_max]
        ], dtype=torch.float32)

        self.categorical_dims = [0, 1, 2, 3]
        self.continuous_dims = [4, 5]

        print(f"\nUpdated optimization bounds:")
        print(f"Supports: [{self.bounds[0, 0]:.1f}, {self.bounds[1, 0]:.1f}] ({n_supports} unique)")
        print(f"Amine 1: [{self.bounds[0, 1]:.1f}, {self.bounds[1, 1]:.1f}] ({n_amines1} unique)")
        print(f"Amine 2: [{self.bounds[0, 2]:.1f}, {self.bounds[1, 2]:.1f}] ({n_amines2} unique)")
        print(f"Amine 3: [{self.bounds[0, 3]:.1f}, {self.bounds[1, 3]:.1f}] ({n_amines3} unique)")
        print(f"MW_Mn: [{self.bounds[0, 4]:.1f}, {self.bounds[1, 4]:.1f}] g/mol")
        print(f"Organic Content: [{self.bounds[0, 5]:.1f}, {self.bounds[1, 5]:.1f}]%")

        self._update_training_data()

        print(f"Updated bounds and conditions:")
        print(f"  Target conditions: {self.target_conditions}")
        print(f"  Categorical bounds: {self.user_categorical_bounds}")
        print(f"  Continuous bounds: {self.user_continuous_bounds}")


class InteractiveCatalystOptimizer:
    def __init__(self,
                 data_path: str,
                 target_conditions: Optional[Dict] = None,
                 categorical_bounds: Optional[Dict] = None,
                 continuous_bounds: Optional[Dict] = None):
        self.data_path = data_path
        self.target_conditions = target_conditions or {}
        self.categorical_bounds = categorical_bounds or {}
        self.continuous_bounds = continuous_bounds or {}

        self.iteration = 0
        self.best_capacity = 0.0
        self.best_config = None

    def run(self):
        """Main interactive optimization loop"""
        print("\n" + "=" * 70)
        print("CATALYST OPTIMIZATION SYSTEM - INTERACTIVE MODE")
        print("=" * 70)

        try:
            data_processor = HistoricalDataProcessor(self.data_path)
            self.bo_system = CatalystBOWithHistory(
                data_processor,
                self.target_conditions,
                self.categorical_bounds,
                self.continuous_bounds
            )
        except Exception as e:
            print(f"Error initializing BO system: {e}")
            return

        while True:
            self.iteration += 1
            print(f"\n{'=' * 70}")
            print(f"ITERATION {self.iteration}")
            print(f"{'=' * 70}")

            self._display_status()
            action = self._get_user_action()

            if action == '1':
                self._generate_and_display_candidates()
            elif action == '2':
                self._record_experiment()
            elif action == '3':
                self._view_history()
            elif action == '4':
                self._export_data()
            elif action == '5':
                self._toggle_data_mode()
            elif action == '6':
                self._view_best_result()
            elif action == '7':
                continue
            elif action == '0':
                print("\nExiting optimization system. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def _display_status(self):
        print(f"\nCurrent Status:")
        print(f"  Iteration: {self.iteration}")
        print(f"  Real experiments: {len(self.bo_system.real_experiments_df)}")
        print(f"  Total training data: {len(self.bo_system.train_X)}")
        if len(self.bo_system.train_X) > 0:
            print(f"  Current best capacity: {self.bo_system.train_Y.max().item():.4f} mmol/g")
        else:
            print(f"  Current best capacity: N/A")
        print(f"  Data mode: {'Real experiments only' if self.bo_system.use_real_data_only else 'Historical + Real'}")

    def _get_user_action(self) -> str:
        print("\nAvailable actions:")
        print("  1. Generate new candidates")
        print("  2. Record experimental result")
        print("  3. View experiment history")
        print("  4. Export data to CSV")
        print("  5. Toggle data mode (Historical+Real ↔ Real only)")
        print("  6. View best result so far")
        print("  7. Continue to next iteration")
        print("  0. Exit")
        return input("\nEnter your choice (0-7): ").strip()

    def _generate_and_display_candidates(self):
        while True:
            try:
                n_candidates = int(input("\nHow many candidates to generate? (1-5, default 3): ") or "3")
                if 1 <= n_candidates <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")

        candidates = self.bo_system.generate_new_candidates(n_candidates)

        if not candidates:
            print("Failed to generate candidates.")
            return

        print(f"\nGenerated {len(candidates)} candidate(s):")
        for i, candidate in enumerate(candidates):
            print(f"\nCandidate {i + 1}:")
            print(f"  Support: {candidate['Support']}")
            print(f"  Amine 1: {candidate['Amine_1_or_Additive_1']}")
            print(f"  Amine 2: {candidate['Amine_2_or_Additive_2']}")
            print(f"  Amine 3: {candidate['Amine_3_or_Additive_3']}")
            print(f"  MW_Mn: {candidate['MW_Mn_g_mol']:.4f} g/mol")
            print(f"  Organic Content: {candidate['Organic_Content_pct']:.4f}%")
            print(f"  Predicted CO2 Capacity: {candidate['Predicted_CO2_Capacity_mmol_g']:.4f} mmol/g")
            print(f"  Is New Configuration: {candidate['Is_New']}")

        self.current_candidates = candidates

    def _record_experiment(self):
        if not hasattr(self, 'current_candidates') or not self.current_candidates:
            print("\nNo candidates available. Please generate candidates first.")
            return

        print("\nSelect candidate to record result for:")
        for i in range(len(self.current_candidates)):
            print(f"  {i + 1}. Candidate {i + 1}")

        try:
            choice = int(input("\nEnter candidate number: ")) - 1
            if 0 <= choice < len(self.current_candidates):
                candidate = self.current_candidates[choice]
            else:
                print("Invalid choice.")
                return
        except ValueError:
            print("Please enter a valid number.")
            return

        print(f"\nRecording experiment for Candidate {choice + 1}:")
        for key, value in candidate.items():
            if key not in ['Predicted_CO2_Capacity_mmol_g', 'Is_New']:
                print(f"  {key}: {value}")

        while True:
            try:
                actual_capacity = float(input("\nEnter actual CO2 capacity (mmol/g): "))
                if actual_capacity >= 0:
                    break
                else:
                    print("Capacity must be non-negative.")
            except ValueError:
                print("Please enter a valid number.")

        notes = input("Enter any notes (optional): ").strip()

        self.bo_system.add_experimental_result(candidate, actual_capacity)

        if actual_capacity > self.best_capacity:
            self.best_capacity = actual_capacity
            self.best_config = candidate.copy()
            self.best_config['Actual_CO2_Capacity_mmol_g'] = actual_capacity
            print(f"\n🎉 New best capacity: {actual_capacity:.4f} mmol/g!")

        if 'Predicted_CO2_Capacity_mmol_g' in candidate:
            predicted = candidate['Predicted_CO2_Capacity_mmol_g']
            error = abs(actual_capacity - predicted)
            print(f"Prediction error: {error:.4f} mmol/g")

    def _view_history(self):
        if len(self.bo_system.real_experiments_df) == 0:
            print("\nNo real experiments recorded yet.")
            return

        print(f"\nReal Experiment History ({len(self.bo_system.real_experiments_df)} experiments):")
        print("-" * 100)

        recent_df = self.bo_system.real_experiments_df.tail(10)
        for idx, row in recent_df.iterrows():
            print(f"\nExperiment {idx + 1}:")
            print(f"  Support: {row['Support']}")
            print(f"  Amine 1: {row['Amine_1_or_Additive_1']}")
            print(f"  Amine 2: {row['Amine_2_or_Additive_2']}")
            print(f"  Amine 3: {row['Amine_3_or_Additive_3']}")
            print(f"  MW_Mn: {row['MW_Mn_g_mol']:.4f} g/mol")
            print(f"  Organic Content: {row['Organic_Content_pct']:.4f}%")
            print(f"  CO2 Capacity: {row['CO2_Capacity_mmol_g']:.4f} mmol/g")

    def _export_data(self):
        filename = input("\nEnter filename for export (default: catalyst_experiments.csv): ") or "catalyst_experiments.csv"
        try:
            combined_df = pd.concat([self.bo_system.historical_df, self.bo_system.real_experiments_df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
            print(f"Data exported to {filename}")
            print(f"Total records: {len(combined_df)}")
        except Exception as e:
            print(f"Error exporting data: {e}")

    def _toggle_data_mode(self):
        current_mode = self.bo_system.use_real_data_only
        new_mode = not current_mode
        self.bo_system.use_real_data_only = new_mode
        mode_name = "real experiments only" if new_mode else "historical + real experiments"
        print(f"\nSwitched to {mode_name} mode for GP fitting.")

    def _view_best_result(self):
        best_in_data = self.bo_system.train_Y.max().item() if len(self.bo_system.train_Y) > 0 else 0.0
        print(f"\nBest Results:")
        print(f"  Best in training data: {best_in_data:.4f} mmol/g")

        if self.best_config:
            print(f"\nBest real experiment so far:")
            print(f"  Support: {self.best_config['Support']}")
            print(f"  Amine 1: {self.best_config['Amine_1_or_Additive_1']}")
            print(f"  Amine 2: {self.best_config['Amine_2_or_Additive_2']}")
            print(f"  Amine 3: {self.best_config['Amine_3_or_Additive_3']}")
            print(f"  MW_Mn: {self.best_config['MW_Mn_g_mol']:.4f} g/mol")
            print(f"  Organic Content: {self.best_config['Organic_Content_pct']:.4f}%")
            print(f"  CO2 Capacity: {self.best_capacity:.4f} mmol/g")
        else:
            print("\nNo real experiments recorded yet.")


def main():
    """Main interactive optimization program"""
    target_conditions = {
        'Relative_Humidity_pct': 0,
        'CO2_Concentration_vol_pct': 0.04,
        'Flow_Rate_mL_min': 100.0,
        'Adsorption_Temperature_C': 25.0,
        'CO2_Test_Method': 'BET'
    }

    categorical_bounds = {
        "Support": ['SBA-15', 'NS', 'MCM-41'],
        "Amine_1_or_Additive_1": ['BPEI', 'TEPA', 'DEA'],
        "Amine_2_or_Additive_2": ['No', 'DEA', 'CTAB', 'P123', 'PC'],
        "Amine_3_or_Additive_3": ['No', 'CTAC'],
    }

    continuous_bounds = {
        "MW_Mn_g_mol": (0, 10000),
        "Organic_Content_pct": (0, 100)
    }

    optimizer = InteractiveCatalystOptimizer(
        data_path='data/historical_experiments.csv',
        target_conditions=target_conditions,
        categorical_bounds=categorical_bounds,
        continuous_bounds=continuous_bounds
    )

    optimizer.run()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()