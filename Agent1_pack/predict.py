
import os
import joblib
import numpy as np
import pandas as pd

from sklearn import set_config
from agent1_components import (
    MissingIndicatorAdder,
    FeatureEngineer,
    FoldWeightedXGBClassifier
)


set_config(transform_output="pandas")

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    PACKAGE_DIR, "agent1_validated_package.pkl"
)


class Agent1FailureDetector:

    def __init__(self, package_path=MODEL_PATH):
        package = joblib.load(package_path)

        self.model = package["model"]
        self.type_map = package["type_map"]
        self.threshold = float(package["cost_threshold"])
        self.qhat = package["conformal_qhats"]
        self.alpha = package["conformal_alpha"]
        self.calibration_method = package["calibration_method"]

        if not isinstance(self.qhat, dict):
            raise ValueError("Expected class-conditional conformal qhats.")

        self.qhat = {int(k): float(v) for k, v in self.qhat.items()}

        if set(self.qhat) != {0, 1}:
            raise ValueError("Conformal qhats must contain classes 0 and 1.")

    def predict(self, machine_data):

        data = machine_data.copy()

        machine_type = data["Type"]

        if isinstance(machine_type, str):
            machine_type = machine_type.upper()

            if machine_type not in self.type_map:
                raise ValueError(f"Unknown machine type: {machine_type}")

            machine_type = self.type_map[machine_type]

        else:
            machine_type = int(machine_type)

            if machine_type not in self.type_map.values():
                raise ValueError(f"Unknown machine type: {machine_type}")

        X_new = pd.DataFrame([{
            "Type": machine_type,
            "Air_temperature_K": data.get("Air_temperature_K", np.nan),
            "Process_temperature_K": data.get("Process_temperature_K", np.nan),
            "Rotational_speed_rpm": data.get("Rotational_speed_rpm", np.nan),
            "Torque_Nm": data.get("Torque_Nm", np.nan),
            "Tool_wear_min": data.get("Tool_wear_min", np.nan)
        }])

        # Preprocessing is performed by the saved pipeline
        p1 = float(self.model.predict_proba(X_new)[0, 1])
        p0 = 1.0 - p1

        failure_detected = bool(p1 >= self.threshold)

        # Class-conditional Mondrian conformal prediction
        prediction_set = []

        if (1.0 - p0) <= self.qhat[0]:
            prediction_set.append("NO FAILURE")

        if (1.0 - p1) <= self.qhat[1]:
            prediction_set.append("FAILURE")

        uncertain = len(prediction_set) != 1

        if len(prediction_set) == 1:
            uncertainty_state = "SINGLETON"
        elif len(prediction_set) == 0:
            uncertainty_state = "EMPTY_SET"
        else:
            uncertainty_state = "AMBIGUOUS_SET"

        agent2_payload = None

        if failure_detected:
            agent2_payload = {
                **X_new.iloc[0].to_dict(),
                "agent1_failure_probability": round(p1, 6),
                "agent1_uncertain": uncertain,
                "agent1_uncertainty_state": uncertainty_state,
                "agent1_prediction_set": prediction_set
            }

        return {
            "prediction": "FAILURE" if failure_detected else "NO FAILURE",
            "failure_probability": round(p1, 6),
            "failure_detected": failure_detected,
            "threshold": round(self.threshold, 6),
            "prediction_set": prediction_set,
            "uncertain": uncertain,
            "uncertainty_state": uncertainty_state,
            "conformal_alpha": self.alpha,
            "conformal_qhat": self.qhat,
            "calibration_method": self.calibration_method,
            "send_to_agent2": failure_detected,
            "agent2_payload": agent2_payload
        }


_agent = None


def predict_failure(machine_data):
    global _agent

    if _agent is None:
        _agent = Agent1FailureDetector()

    return _agent.predict(machine_data)
