# Agent 1 — Predictive Maintenance Failure Detection

**Calibrated binary failure prediction, uncertainty quantification, and cost-sensitive decision support**

Agent 1 is a machine-learning component for predictive maintenance of industrial equipment. It estimates the probability of machine failure from sensor measurements and returns:

- A calibrated failure probability from an XGBoost-based pipeline.
- A binary decision using a saved cost-sensitive threshold.
- A class-conditional (Mondrian) conformal prediction set and uncertainty state.
- A routing flag and payload for a downstream **Agent 2** when failure is detected.

**Scope:** Agent 1 predicts whether a failure is detected; it does **not** identify the type of failure. Diagnosis belongs to Agent 2.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`pipeline/`](pipeline/) | Model development, validation, and deployment-generation notebook |
| [`Agent1_pack/`](Agent1_pack/) | Standalone inference package, including the trained model |
| [`Data/`](Data/) | Dataset used by the project |
| [`Agent1_pack/README.md`](Agent1_pack/README.md) | Detailed installation, inference, and output documentation |

The deployment package contains six files: `agent1_components.py`, `agent1_validated_package.pkl`, `predict.py`, `test_agent1.py`, `requirements.txt`, and `README.md`.

## Quick start

These instructions use the files in `Agent1_pack/` after cloning the repository:

```bash
git clone https://github.com/melinahou/Agent_1.git
cd Agent_1
python -m pip install -r Agent1_pack/requirements.txt
python Agent1_pack/test_agent1.py
```

The smoke test checks that the packaged model loads and produces an inference result. It is **not** a test of predictive accuracy.

### Run a prediction

From the repository root:

```python
import sys
sys.path.insert(0, "Agent1_pack")

from predict import predict_failure

machine_data = {
    "Type": "M",
    "Air_temperature_K": 300.5,
    "Process_temperature_K": 311.2,
    "Rotational_speed_rpm": 1350,
    "Torque_Nm": 65.0,
    "Tool_wear_min": 200,
}

result = predict_failure(machine_data)
print(result)
```

Machine types may be specified as `L`, `M`, `H` or the corresponding codes `0`, `1`, `2`. The saved preprocessing pipeline handles missing numeric sensor values and constructs engineered features internally.

For step-by-step **Google Colab** instructions, see [the package README](Agent1_pack/README.md).

## Prediction output

The public function `predict_failure(machine_data)` returns a dictionary with:

| Key | Meaning |
| --- | --- |
| `prediction` | `FAILURE` or `NO FAILURE` based on the cost threshold |
| `failure_probability` | Calibrated failure probability |
| `failure_detected` | Boolean cost-sensitive decision |
| `threshold` | Threshold saved in the model package |
| `prediction_set` | Mondrian conformal set: zero, one, or both class labels |
| `uncertain`, `uncertainty_state` | Whether the set is empty, ambiguous, or a singleton |
| `conformal_alpha`, `conformal_qhat` | Conformal configuration and class-specific quantiles |
| `calibration_method` | Saved probability calibration method |
| `send_to_agent2`, `agent2_payload` | Downstream routing decision and optional payload |

**Important:** The cost-sensitive decision and the conformal prediction set answer different questions. A failure can be flagged by the decision threshold even when the conformal set is ambiguous or empty.

## Methodology

The development notebook covers:

1. Data preparation, feature engineering, and group-aware validation ( 1,2,3,4.2).
2. Class imbalance handling, baseline models, and hyperparameter optimization ( 4,5,6).
3. Probability calibration and reliability assessment (7).
4. Cost-sensitive threshold selection (8,10).
5. Mondrian split-conformal uncertainty estimation (9).
6. Independent test evaluation and publication-oriented figures (10,11,13).
7. Model serialization, deployment, and notebook-versus-deployment consistency checks (12,14,15,16).

The deployed model uses pipeline-based preprocessing and feature engineering, XGBoost classification, isotonic calibration, and saved cost/conformal parameters. The notebook contains the experimental methodology and evaluation outputs; the standalone package is for inference.


### Deployment Consistency and Validation Checks

To verify the consistency of the deployment implementation with the original notebook, a series of tests was performed to ensure that the standalone Agent 1 produces results consistent with the original notebook.

The tests verify:

- **Prediction consistency:** Agreement between the notebook and deployment model in predicted failure probabilities.
- **Decision consistency:** Agreement in cost-sensitive failure decisions and Agent 2 routing.
- **Uncertainty consistency:** Agreement in Mondrian conformal prediction sets.
- **Missing data handling:** Correct operation when numerical sensor values are missing.
- **Class-balanced testing:** Consistency across a test sample containing equal numbers of failure and non-failure cases.
- **Input compatibility:** Correct handling of machine types `L`, `M`, and `H`, including their numerical representations.

All performed deployment consistency tests passed without prediction-set or decision mismatches.

**Note:** These tests confirm that the deployed implementation reproduces the notebook's behavior on the tested samples. Predictive performance and conformal coverage are evaluated separately in the notebook using the independent test set.

