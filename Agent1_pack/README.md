# Agent 1 — Predictive Maintenance
## Calibrated Failure Prediction and Uncertainty Estimation

Agent 1 predicts whether an industrial machine is at risk of failure.
It returns a calibrated failure probability, a cost-sensitive binary
decision, and a class-conditional conformal prediction set.

Agent 1 does not diagnose the type of failure. When a failure is
detected, it prepares a payload for Agent 2.

## 1. Package Contents

The deployment ZIP contains exactly six files:

- `agent1_components.py` — Custom preprocessing and model components.
- `agent1_validated_package.pkl` — Saved calibrated model and decision parameters.
- `predict.py` — Standalone prediction interface.
- `test_agent1.py` — Deployment smoke test.
- `requirements.txt` — Python dependencies.
- `README.md` — Installation and usage instructions.

The original training notebook and dataset are not required for inference.

## 2. Installation in Google Colab

Start a new Google Colab notebook and upload `agent1_package.zip`.

```python
from google.colab import files
files.upload()
```

Extract the archive:

```python
import zipfile

with zipfile.ZipFile("/content/agent1_package.zip") as archive:
    archive.extractall("/content/agent1_package")
```

Install dependencies:

```python
%pip install -r /content/agent1_package/requirements.txt
```

Restart the runtime if necessary.

## 3. Run the Deployment Smoke Test

Execute:

```python
!python /content/agent1_package/test_agent1.py
```

A successful execution prints the prediction output and a success message.
If inference fails, the script exits with an error.

## 4. Run a Prediction

Execute the following code in a Colab cell:

```python
import sys

sys.path.insert(0, "/content/agent1_package")

from predict import predict_failure

sample_data = {
    "Type": "M",
    "Air_temperature_K": 300.5,
    "Process_temperature_K": 311.2,
    "Rotational_speed_rpm": 1350,
    "Torque_Nm": 65.0,
    "Tool_wear_min": 200
}

result = predict_failure(sample_data)

for key, value in result.items():
    print(f"{key}: {value}")
```

The exact predicted probability depends on the saved model and
compatible software environment.

## 5. Input Features

| Feature | Description | Unit |
|---|---|---|
| Type | Machine type: L, M, H or 0, 1, 2 | Categorical |
| Air_temperature_K | Ambient air temperature | K |
| Process_temperature_K | Process temperature | K |
| Rotational_speed_rpm | Rotational speed | rpm |
| Torque_Nm | Applied torque | Nm |
| Tool_wear_min | Accumulated tool wear | min |

Machine type mapping:
- L = 0
- M = 1
- H = 2

The machine type is required and must be valid.

Missing numerical sensor values are accepted and processed by
the saved preprocessing pipeline. Invalid machine types raise an error.

Derived features are computed internally. Users should provide
the original sensor measurements rather than engineered features.

## 6. Prediction Output

The `predict_failure()` function returns a dictionary containing:

- `prediction`: Cost-sensitive binary decision, FAILURE or NO FAILURE.
- `failure_probability`: Calibrated probability of failure.
- `failure_detected`: Boolean cost-sensitive decision.
- `threshold`: Saved cost-based decision threshold.
- `prediction_set`: Mondrian conformal prediction set.
- `uncertain`: Whether the prediction set is not a singleton.
- `uncertainty_state`: SINGLETON, EMPTY_SET or AMBIGUOUS_SET.
- `conformal_alpha`: Configured conformal miscoverage level.
- `conformal_qhat`: Class-conditional conformal quantiles.
- `calibration_method`: Probability calibration method.
- `send_to_agent2`: Whether to route the observation to Agent 2.
- `agent2_payload`: Data prepared for Agent 2, or None.

## 7. Decision Logic and Uncertainty

The binary failure decision uses the saved cost-sensitive threshold:

    failure_detected = failure_probability >= threshold

The conformal prediction set is calculated separately.

Possible prediction sets include:

- `["NO FAILURE"]`: Only the no-failure class is included.
- `["FAILURE"]`: Only the failure class is included.
- `["NO FAILURE", "FAILURE"]`: Both classes are included.
- `[]`: Neither class is included.

`SINGLETON` indicates that exactly one class is included.
`AMBIGUOUS_SET` indicates that both classes are included.
`EMPTY_SET` indicates that neither class is included.

The conformal prediction set is not the same as the
cost-sensitive binary decision.

The uncertainty flag is True when the set is ambiguous or empty.

## 8. Agent 2 Integration

If `failure_detected` is True:

- `send_to_agent2` is True.
- `agent2_payload` contains the machine measurements,
  failure probability, and uncertainty information.

Otherwise:

- `send_to_agent2` is False.
- `agent2_payload` is None.

Agent 2 is responsible for downstream failure diagnosis.
Agent 1 does not predict the failure type.

## 9. Model and Validation Notes

The saved model uses:

- XGBoost-based binary classification.
- Pipeline-based preprocessing and feature engineering.
- Isotonic probability calibration.
- Cost-sensitive decision thresholding.
- Class-conditional Mondrian conformal prediction.

Model development and scientific evaluation are documented
in the original training notebook.

The standalone package is intended for inference.
Its smoke test verifies execution, not scientific performance.

## 10. Important Notes

- Keep all six deployment files together.
- Do not retrain the model during inference.
- Do not manually calculate engineered features.
- Do not copy the entire `predict.py` module into a Colab cell.
- Import `predict_failure` from the module as shown above.
- Use compatible dependency versions for reproducible results.
- Predictions are statistical estimates, not guarantees of machine failure.
