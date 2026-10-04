
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
# Feature Engineering (Πρέπει να είναι απολύτως ταυτόσημο με το training)
# ============================================================================

def engineer_features(X):
    X = X.copy()

    # Υπολογισμός ισχύος σε Watt
    X['Power_W'] = (
        X['Torque_Nm']
        * (
            X['Rotational_speed_rpm']
            * 2 * np.pi / 60
        )
    )

    # Υπολογισμός διαφοράς θερμοκρασίας σε Kelvin
    X['Temp_diff_K'] = (
        X['Process_temperature_K']
        - X['Air_temperature_K']
    )

    return X


# Ορισμός στο __main__ ώστε να επιτρέπεται το σωστό deserialization του fitted pipeline
import __main__
__main__.engineer_features = engineer_features


# ============================================================================
# Paths
# ============================================================================

if '__file__' in globals():
    SCRIPT_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )
else:
    # When code is copied directly into a Colab/Jupyter cell
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    'agent1_validated_package.pkl'
)

# Extra fallback for Colab
if not os.path.exists(MODEL_PATH):

    colab_path = (
        '/content/agent1_package/'
        'agent1_validated_package.pkl'
    )

    if os.path.exists(colab_path):
        MODEL_PATH = colab_path

# ============================================================================
# Agent 1
# ============================================================================

class Agent1FailureDetector:
    """
    Agent 1 — Standalone Binary Failure Detection Inference Engine.
    Φορτώνει το επικυρωμένο πακέτο και παρέχει calibrated προβλέψεις,
    cost-sensitive αποφάσεις και conformal prediction sets.
    """
    def __init__(self, package_path=MODEL_PATH):
        package = joblib.load(package_path)

        self.model = package['model']
        self.type_map = package['type_map']
        self.threshold = package['cost_threshold']
        self.qhat = package['conformal_qhat']
        self.alpha = package['conformal_alpha']
        self.calibration_method = package['calibration_method']


    def predict(self, machine_data):

        data = machine_data.copy()

        # 1. Κωδικοποίηση τύπου μηχανής (Type mapping)
        machine_type = data['Type']

        if isinstance(machine_type, str):

            machine_type = machine_type.upper()

            if machine_type not in self.type_map:
                raise ValueError(
                    f"Unknown Type: {machine_type}. "
                    f"Expected {list(self.type_map.keys())}"
                )

            machine_type = self.type_map[
                machine_type
            ]

        else:
            machine_type = int(
                machine_type
            )


        # 2. Δημιουργία τυποποιημένου DataFrame εισόδου
        X_new = pd.DataFrame([{
            'Type': machine_type,
            'Air_temperature_K': data.get('Air_temperature_K', np.nan),
            'Process_temperature_K': data.get('Process_temperature_K', np.nan),
            'Rotational_speed_rpm': data.get('Rotational_speed_rpm', np.nan),
            'Torque_Nm': data.get('Torque_Nm', np.nan),
            'Tool_wear_min': data.get('Tool_wear_min', np.nan)
        }])

        # 3. Υπολογισμός calibrated πιθανότητας αποτυχίας
        failure_probability = float(
            self.model.predict_proba(X_new)[0, 1]
        )

        # 4. Cost-sensitive δυαδική απόφαση βάσει του βέλτιστου κατωφλίου (threshold)
        failure_detected = bool(
            failure_probability >= self.threshold
        )

        # 5. Κατασκευή conformal prediction set και έλεγχος αβεβαιότητας
        class_probs = np.array([
            1.0 - failure_probability,
            failure_probability
        ])

        included = (
            1.0 - class_probs
            <= self.qhat
        )

        prediction_set = []

        if included[0]:
            prediction_set.append(
                'NO FAILURE'
            )

        if included[1]:
            prediction_set.append(
                'FAILURE'
            )


        uncertain = (
            len(prediction_set) != 1
        )

        if len(prediction_set) == 1:
            uncertainty_state = 'SINGLETON'

        elif len(prediction_set) == 0:
            uncertainty_state = 'EMPTY_SET'

        else:
            uncertainty_state = 'AMBIGUOUS_SET'


       # 6. Δημιουργία δομημένου payload για διασύνδεση με τον Agent 2 (εφόσον ανιχνευθεί σφάλμα)
        agent2_payload = None
        if failure_detected:
            agent2_payload = {
                'Type': machine_type,
                'Air_temperature_K': data.get('Air_temperature_K', np.nan),
                'Process_temperature_K': data.get('Process_temperature_K', np.nan),
                'Rotational_speed_rpm': data.get('Rotational_speed_rpm', np.nan),
                'Torque_Nm': data.get('Torque_Nm', np.nan),
                'Tool_wear_min': data.get('Tool_wear_min', np.nan),
                'agent1_failure_probability': round(failure_probability, 6),
                'agent1_uncertain': uncertain,
                'agent1_uncertainty_state': uncertainty_state,
                'agent1_prediction_set': prediction_set
            }

        # 7. Τελικό λεξικό αποτελεσμάτων παραγωγής
        return {
            'prediction': 'FAILURE' if failure_detected else 'NO FAILURE',
            'failure_probability': round(failure_probability, 6),
            'failure_detected': failure_detected,
            'threshold': round(self.threshold, 6),
            'prediction_set': prediction_set,
            'uncertain': uncertain,
            'uncertainty_state': uncertainty_state,
            'conformal_alpha': self.alpha,
            'conformal_qhat': round(self.qhat, 6),
            'calibration_method': self.calibration_method,
            'send_to_agent2': failure_detected,
            'agent2_payload': agent2_payload
        }


# ============================================================================
# Public interface
# ============================================================================

_agent = None


def predict_failure(machine_data):

    """
    Δημόσια συνάρτηση διασύνδεσης για εύκολη κλήση της πρόβλεψης
    χωρίς την ανάγκη manual χειρισμού της κλάσης.
    """
    global _agent

    if _agent is None:
        _agent = Agent1FailureDetector()

    return _agent.predict(
        machine_data
    )

# Δοκιμή πρόβλεψης με τυπικά δεδομένα αισθητήρων μηχανής
result = predict_failure({
    'Type': 'M',
    'Air_temperature_K': 300.5,
    'Process_temperature_K': 311.2,
    'Rotational_speed_rpm': 1350,
    'Torque_Nm': 65.0,
    'Tool_wear_min': 200
})

print("Sample prediction result:")
print(result)
