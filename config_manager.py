import json
import os
import pickle
from sklearn.preprocessing import LabelEncoder

class ConfigManager:
    def __init__(self, config_path=None):
        self.config_path = config_path or "config/default_config.json"
        self.config = None
        self.label_encoders = None

        self.load_config()
        self.load_encoders()

    def load_config(self, config_path=None):
        if config_path:
            self.config_path = config_path

        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"Loaded configuration from {self.config_path}")
        else:
            print("Configuration file not found. Using default configuration.")
            self.config = self._default_config()

        self.validate_config()
        return self.config

    def _default_config(self):
        return {
            "numerical_columns": ["Organic_Content_pct"],
            "categorical_columns": [
                "Support",
                "Amine_1_or_Additive_1",
                "Amine_2_or_Additive_2",
                "Amine_3_or_Additive_3"
            ],
            "target_columns": ["CO2_Capacity_mmol_g"],
            "bounds": {
                "Organic_Content_pct": [0, 100]
            },
            "optimization": {
                "n_candidates": 5,
                "n_restarts": 10,
                "raw_samples": 1000
            }
        }

    def validate_config(self):
        required_keys = ['numerical_columns', 'categorical_columns', 'target_columns', 'bounds']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        if not isinstance(self.config['target_columns'], list):
            self.config['target_columns'] = [self.config['target_columns']]
        print("Configuration validated successfully")
        return True

    def load_encoders(self):
        if self.config is None:
            self.label_encoders = {}
            return
        encoders_path = self.config.get('encoders_load_path', '')
        if encoders_path and os.path.exists(encoders_path):
            try:
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                print(f"Loaded label encoders from {encoders_path}")
                for col in self.config['categorical_columns']:
                    if col not in self.label_encoders:
                        print(f"Warning: No encoder found for categorical column {col}")
            except Exception as e:
                print(f"Error loading label encoders, creating new ones: {e}")
                self.label_encoders = {}
        else:
            print("No encoder path specified or file not found, will create new encoders")
            self.label_encoders = {}

    def save_encoders(self, label_encoders, save_path=None):
        if not save_path:
            save_path = self.config.get('encoders_save_path', 'label_encoders.pkl') if self.config else 'label_encoders.pkl'
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(label_encoders, f)
            print(f"Saved label encoders to {save_path}")
            if self.config:
                self.config['encoders_load_path'] = save_path
                self.config['encoders_save_path'] = save_path
            return True
        except Exception as e:
            print(f"Error saving label encoders: {e}")
            return False

    def save_config(self, save_path=None):
        if save_path:
            self.config_path = save_path
        if not self.config_path or self.config is None:
            print("No config path specified or no config to save")
            return False
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def get_all_feature_columns(self):
        return self.config['numerical_columns'] + self.config['categorical_columns']

    def get_predictor(self):
        return self.config.get('predictor', "random_forest")

    def get_feature_type(self, column_name):
        if column_name in self.config['numerical_columns']:
            return 'numerical'
        elif column_name in self.config['categorical_columns']:
            return 'categorical'
        else:
            return None

    def get_bounds_for_feature(self, feature_name):
        bounds = self.config.get('bounds', {})
        return bounds.get(feature_name)

    def get_optimization_param(self, param_name, default=None):
        optimization = self.config.get('optimization', {})
        return optimization.get(param_name, default)

    def get_categorical_encoder(self, column_name):
        if self.label_encoders and column_name in self.label_encoders:
            return self.label_encoders[column_name]
        return None

    def set_categorical_encoder(self, column_name, encoder):
        if self.label_encoders is None:
            self.label_encoders = {}
        self.label_encoders[column_name] = encoder

    def initialize_encoders_from_data(self, data):
        if self.label_encoders is None:
            self.label_encoders = {}
        for col in self.config['categorical_columns']:
            if col in data.columns and col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                unique_values = data[col].dropna().unique()
                if len(unique_values) > 0:
                    self.label_encoders[col].fit(unique_values)
                    print(f"Created encoder for {col} with {len(unique_values)} classes")