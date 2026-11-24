
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
    PACKAGE_DIR, "agent1_raw_mondrian_development.pkl"
)


class Agent1FailureDetector:

    def __init__(self, package_path=MODEL_PATH):
        package = joblib.load(package_path)

        self.model = package["model"]
        self.type_map = package["type_map"]
        self.threshold = float(package["cost_threshold"])
        self.qhat = {
            int(k): float(v)
            for k, v in package["conformal_qhats"].items()
        }
        self.alpha = float(package["conformal_alpha"])

        assert set(self.qhat) == {0, 1}

    def predict(self, machine_data):

        if not isinstance(machine_data, dict):
            raise ValueError("Machine data must be a dictionary.")

        data = machine_data.copy()

        # Validate machine type
        machine_type = data.get("Type")

        if isinstance(machine_type, str):
            machine_type = machine_type.strip().upper()

            if machine_type not in self.type_map:
                raise ValueError("Unknown machine type.")

            machine_type = self.type_map[machine_type]

        if (
            isinstance(machine_type, (bool, np.bool_))
            or not isinstance(
                machine_type,
                (int, float, np.integer, np.floating)
            )
            or machine_type not in self.type_map.values()
        ):
            raise ValueError("Type must be L, M, H, 0, 1 or 2.")

        # Validate sensor measurements
        sensor_fields = [
            "Air_temperature_K",
            "Process_temperature_K",
            "Rotational_speed_rpm",
            "Torque_Nm",
            "Tool_wear_min"
        ]

        validated_data = {"Type": int(machine_type)}

        for field in sensor_fields:

            value = data.get(field, np.nan)

            if value is None:
                value = np.nan

            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value,
                    (int, float, np.integer, np.floating)
                )
            ):
                raise ValueError(
                    f"{field} must be numeric or missing."
                )

            value = float(value)

            if np.isinf(value):
                raise ValueError(
                    f"{field} cannot be infinite."
                )

            validated_data[field] = value

        # Reject observations with no sensor information
        if all(
            np.isnan(validated_data[field])
            for field in sensor_fields
        ):
            raise ValueError("All sensor measurements are missing.")

        X_new = pd.DataFrame([validated_data])

        p = float(self.model.predict_proba(X_new)[0, 1])

        prediction_set = []
        if p <= self.qhat[0]:
            prediction_set.append("NO FAILURE")
        if 1 - p <= self.qhat[1]:
            prediction_set.append("FAILURE")

        failure_detected = p >= self.threshold

        if len(prediction_set) == 0:
            status = "REVIEW_REQUIRED"
        elif len(prediction_set) == 2:
            status = "UNCERTAIN"
        else:
            status = "FAILURE" if prediction_set[0] == "FAILURE" else "HEALTHY"

        return {
            "failure_probability": p,
            "prediction": "FAILURE" if failure_detected else "NO FAILURE",
            "threshold": self.threshold,
            "prediction_set": prediction_set,
            "uncertainty_status": status,
            "send_to_agent2": bool(failure_detected),
            "conformal_alpha": self.alpha
        }


_agent = None

def predict_failure(machine_data):
    global _agent
    if _agent is None:
        _agent = Agent1FailureDetector()
    return _agent.predict(machine_data)
