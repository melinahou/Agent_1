
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier


# IMPORTANT:
# Same sklearn output configuration used during training
set_config(transform_output="pandas")


# ============================================================================
# Custom Transformers / Feature Engineering
# ============================================================================

class MissingIndicatorAdder:
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X


class FeatureEngineer:
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return engineer_features(X)


def engineer_features(X):
    X = X.copy()
    
    # Υπολογισμός διαφοράς θερμοκρασίας σε Kelvin (μόνο αυτό υπήρχε στο training)
    X['Temp_diff_K'] = (
        X['Process_temperature_K']
        - X['Air_temperature_K']
    )
    
    # Προσθήκη των missing indicators
    X['Air_temperature_K_isnan'] = X['Air_temperature_K'].isna().astype(int)
    X['Process_temperature_K_isnan'] = X['Process_temperature_K'].isna().astype(int)
    
    return X


# Ορισμός στο __main__ ώστε να επιτρέπεται το σωστό deserialization
import __main__
__main__.MissingIndicatorAdder = MissingIndicatorAdder
__main__.FeatureEngineer = FeatureEngineer
__main__.engineer_features = engineer_features


# ============================================================================
# Paths
# ============================================================================

if '__file__' in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, 'agent1_validated_package.pkl')

if not os.path.exists(MODEL_PATH):
    colab_path = '/content/agent1_package/agent1_validated_package.pkl'
    if os.path.exists(colab_path):
        MODEL_PATH = colab_path


# ============================================================================
# Agent 1
# ============================================================================

class Agent1FailureDetector:
    """
    Agent 1 — Standalone Binary Failure Detection Inference Engine.
    """
    def __init__(self, package_path=MODEL_PATH):
        package = joblib.load(package_path)

        self.model = package['model']
        self.type_map = package['type_map']
        self.threshold = package['cost_threshold']
        
        self.qhat = package.get('conformal_qhats', package.get('conformal_qhat'))
        self.alpha = package.get('conformal_alpha', 0.10)
        self.calibration_method = package.get('calibration_method', 'isotonic')

    def predict(self, machine_data):
        data = machine_data.copy()

        # 1. Κωδικοποίηση τύπου μηχανής
        machine_type = data['Type']
        if isinstance(machine_type, str):
            machine_type = machine_type.upper()
            if machine_type not in self.type_map:
                raise ValueError(f"Unknown Type: {machine_type}. Expected {list(self.type_map.keys())}")
            machine_type = self.type_map[machine_type]
        else:
            machine_type = int(machine_type)

        # 2. Δημιουργία τυποποιημένου DataFrame εισόδου
        air_temp = data.get('Air_temperature_K', np.nan)
        proc_temp = data.get('Process_temperature_K', np.nan)

        X_new = pd.DataFrame([{
            'Type': machine_type,
            'Air_temperature_K': air_temp,
            'Process_temperature_K': proc_temp,
            'Rotational_speed_rpm': data.get('Rotational_speed_rpm', np.nan),
            'Torque_Nm': data.get('Torque_Nm', np.nan),
            'Tool_wear_min': data.get('Tool_wear_min', np.nan),
            'Air_temperature_K_isnan': int(pd.isna(air_temp)),
            'Process_temperature_K_isnan': int(pd.isna(proc_temp))
        }])

        # 3. Υπολογισμός πιθανότητας
        failure_probability = float(self.model.predict_proba(X_new)[0, 1])

        # 4. Cost-sensitive δυαδική απόφαση
        failure_detected = bool(failure_probability >= self.threshold)

        # 5. Mondrian Conformal Prediction set
        class_probs = np.array([1.0 - failure_probability, failure_probability])
        
        if isinstance(self.qhat, dict):
            current_qhat = self.qhat.get(machine_type, 0.1)
        else:
            current_qhat = self.qhat

        included = (1.0 - class_probs) <= current_qhat
        prediction_set = []
        if included[0]:
            prediction_set.append('NO FAILURE')
        if included[1]:
            prediction_set.append('FAILURE')

        uncertain = (len(prediction_set) != 1)
        if len(prediction_set) == 1:
            uncertainty_state = 'SINGLETON'
        elif len(prediction_set) == 0:
            uncertainty_state = 'EMPTY_SET'
        else:
            uncertainty_state = 'AMBIGUOUS_SET'

        # 6. Payload για τον Agent 2
        agent2_payload = None
        if failure_detected:
            agent2_payload = {
                'Type': machine_type,
                'Air_temperature_K': air_temp,
                'Process_temperature_K': proc_temp,
                'Rotational_speed_rpm': data.get('Rotational_speed_rpm', np.nan),
                'Torque_Nm': data.get('Torque_Nm', np.nan),
                'Tool_wear_min': data.get('Tool_wear_min', np.nan),
                'agent1_failure_probability': round(failure_probability, 6),
                'agent1_uncertain': uncertain,
                'agent1_uncertainty_state': uncertainty_state,
                'agent1_prediction_set': prediction_set
            }

        return {
            'prediction': 'FAILURE' if failure_detected else 'NO FAILURE',
            'failure_probability': round(failure_probability, 6),
            'failure_detected': failure_detected,
            'threshold': round(self.threshold, 6),
            'prediction_set': prediction_set,
            'uncertain': uncertain,
            'uncertainty_state': uncertainty_state,
            'conformal_alpha': self.alpha,
            'conformal_qhat': self.qhat,
            'calibration_method': self.calibration_method,
            'send_to_agent2': failure_detected,
            'agent2_payload': agent2_payload
        }


# ============================================================================
# Public interface
# ============================================================================

_agent = None


def predict_failure(machine_data):
    global _agent

    if _agent is None:
        _agent = Agent1FailureDetector()

    return _agent.predict(machine_data)
