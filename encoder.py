import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

class FeatureEncoder:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.label_encoders = {}

    def fit_transform_categorical(self, data, categorical_columns):
        transformed_data = data.copy()
        for col in categorical_columns:
            if col in transformed_data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(transformed_data[col].astype(str).unique())
                transformed_data[col] = self.label_encoders[col].transform(
                    transformed_data[col].astype(str)
                )
        return transformed_data

    def transform_categorical(self, data, categorical_columns):
        transformed_data = data.copy()
        for col in categorical_columns:
            if col in transformed_data.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                mask = ~transformed_data[col].astype(str).isin(le.classes_)
                if mask.any():
                    transformed_data.loc[mask, col] = le.classes_[0]
                transformed_data[col] = le.transform(transformed_data[col].astype(str))
        return transformed_data

    def decode_categorical(self, encoded_value, feature_name):
        if feature_name not in self.label_encoders:
            return encoded_value
        encoder = self.label_encoders[feature_name]
        if hasattr(encoder, 'inverse_transform'):
            try:
                return encoder.inverse_transform([int(encoded_value)])[0]
            except (ValueError, IndexError):
                return f"Unknown_{encoded_value}"
        return encoded_value

    def decode_candidate(self, candidate, feature_names=None):
        if feature_names is None:
            feature_names = self.feature_names
        if candidate is None or feature_names is None:
            return None
        decoded = {}
        if len(feature_names) == len(candidate):
            for i, col in enumerate(feature_names):
                if col in self.label_encoders:
                    decoded[col] = self.decode_categorical(candidate[i], col)
                else:
                    decoded[col] = float(candidate[i])
        return decoded

    def encode_candidate(self, candidate_dict, feature_order=None):
        if feature_order is None:
            feature_order = self.feature_names
        if not candidate_dict or not feature_order:
            return None
        encoded = []
        for col in feature_order:
            val = candidate_dict.get(col)
            if val is None:
                encoded.append(0.0)
            elif col in self.label_encoders:
                encoder = self.label_encoders[col]
                # encoder is always a LabelEncoder with transform()
                try:
                    encoded_val = encoder.transform([str(val)])[0]
                except (ValueError, KeyError):
                    # fallback to first class
                    encoded_val = 0
                encoded.append(encoded_val)
            else:
                try:
                    encoded.append(float(val))
                except (TypeError, ValueError):
                    encoded.append(0.0)
        return np.array(encoded)

    def get_encoder_classes(self, column_name):
        if column_name in self.label_encoders:
            encoder = self.label_encoders[column_name]
            if hasattr(encoder, 'classes_'):
                return list(encoder.classes_)
        return []

    def save_encoders(self, file_path='label_encoders.pkl'):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            print(f"Encoders saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving encoders: {e}")
            return False

    def load_encoders(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print(f"Encoders loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading encoders: {e}")
            return False

    def get_bounds(self, categorical_columns=None):
        bounds = {}
        for col in (self.feature_names or []):
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                if hasattr(encoder, 'classes_'):
                    bounds[col] = (0, len(encoder.classes_) - 1)
                else:
                    bounds[col] = (0, 1)
            else:
                bounds[col] = (0, 1)
        return bounds