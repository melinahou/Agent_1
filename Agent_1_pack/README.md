# Agent 1 — Predictive Maintenance

## Overview

Agent 1 estimates the probability of machine failure using
a Plain XGBoost model (scale_pos_weight=1.0).

It returns:

- Raw failure probability (not externally calibrated)
- Cost-sensitive binary decision
- Class-conditional Mondrian conformal prediction set
- Uncertainty status
- Routing flag for Agent 2
- Decision threshold
- Conformal alpha

Agent 1 does not diagnose failure types.

## Package Files

- agent1_components.py
- agent1_raw_mondrian_development.pkl
- predict.py
- reference_cases.json
- test_agent1.py
- requirements.txt
- README.md

## Installation

### Step 1 — Upload and extract the package in Google Colab

Open a new Google Colab notebook and run:

```python
from google.colab import files
import zipfile
import io

uploaded = files.upload()

with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    z.extractall("/content")
```

Select the agent1_package.zip file when prompted.

The package will be extracted to /content/agent1_package.

Training environment: Python 3.13.18.

### Step 2 — Install dependencies

The deployment dependencies listed in requirements.txt
are pinned to their training environment versions.

Run the following command in a new Colab cell:

```python
%pip install -r /content/agent1_package/requirements.txt
```

Restart the session if requested after installation.

### Step 3 — Run the regression test

Run the following command in a new Colab cell:

```python
!python /content/agent1_package/test_agent1.py
```

Expected output:

```text
PASS: All 100 reference cases verified.
```

### Step 4 — Example prediction


    import sys
    sys.path.insert(0, "/content/agent1_package")

    from predict import predict_failure

    machine = {
        "Type": "M",
        "Air_temperature_K": 300.5,
        "Process_temperature_K": 311.2,
        "Rotational_speed_rpm": 1350,
        "Torque_Nm": 65.0,
        "Tool_wear_min": 200
    }

    result = predict_failure(machine)
    print(result)

## Input Validation

Accepted machine types:

- L or 0
- M or 1
- H or 2

Sensor measurements must be numeric or missing.

Missing measurements are handled by the preprocessing
pipeline using missing indicators for temperature
measurements and median imputation.

The following inputs are rejected:

- Invalid machine types
- Missing machine type
- Non-numeric sensor measurements
- Infinite sensor values
- Completely missing sensor measurements

Invalid inputs raise ValueError.

Physical operating limits are not explicitly enforced.

## Output Fields

- failure_probability: Raw XGBoost probability
- prediction: FAILURE or NO FAILURE
- threshold: Selected development threshold stored
  in the model package
- prediction_set: Mondrian conformal prediction set
- uncertainty_status: HEALTHY, FAILURE, UNCERTAIN,
  or REVIEW_REQUIRED
- send_to_agent2: Boolean routing flag
- conformal_alpha: 0.10

## Decision Rules

Binary decision:

    failure_probability >= package cost_threshold

The threshold is selected through development-only
cost optimization using a group-disjoint threshold
selection subset.

Cost ratio:

    C_FN:C_FP = 10:1

Mondrian prediction set:

    Include NO FAILURE if p <= qhat_0
    Include FAILURE if 1-p <= qhat_1

The binary decision and conformal prediction set
are separate outputs.

Uncertainty status:

- Empty prediction set: REVIEW_REQUIRED
- Two-label prediction set: UNCERTAIN
- NO FAILURE singleton: HEALTHY
- FAILURE singleton: FAILURE

## Agent Routing Policy

Agent 2 is activated exclusively when the
failure probability reaches or exceeds the
cost-sensitive decision threshold.

Conformal uncertainty does not directly
activate Agent 2.

Cases marked as REVIEW_REQUIRED or UNCERTAIN
must be handled by a separate review mechanism
at the multi-agent orchestration level,
regardless of the binary failure decision.

This policy preserves the separation between
failure prediction, fault diagnosis and
maintenance decision-making.

The external review mechanism is not implemented
within Agent 1.

## Automated Regression Testing

The package includes reference_cases.json containing
100 fixed test-set reference cases, including:

- Healthy and faulty machines
- Missing temperature measurements
- Cases close to the decision threshold

Run the regression test from the package directory:

    python test_agent1.py

The test verifies:

- Failure probabilities (absolute tolerance: 1e-6)
- Exact failure decisions
- Exact conformal prediction sets
- Uncertainty status
- Agent 2 routing flag
- Decision threshold
- Conformal alpha

In case of an output mismatch, the test reports
the failed case ID, the expected output and the
actual output.

Exit code 0 indicates success.
Exit code 1 indicates a detected mismatch.

The regression test was verified through intentional
threshold corruption, followed by successful
restoration of the original model package.

Additional notebook tests verify:

- Batch and single-sample prediction consistency
- Notebook and deployment probability consistency
- Missing-value prediction consistency
- Machine type mapping consistency
- Rejection of invalid inputs
- Successful restoration after threshold corruption

These tests verify deployment consistency, not
independent predictive performance.

## Random Failure Stress Test

Random failures are excluded from model training.

A development-only stress test was conducted using
random failure observations from development systems.

Results:

- Random failure samples: 12
- Detected random failures: 1
- Detection rate: 8.33%
- Uncertainty rate: 8.33%

These results indicate limited detection capability
for random failures.

The small sample size limits the interpretation
and generalizability of these results.

## Scientific Limitations

- No external probability calibration is applied.

- The selected threshold minimizes empirical cost
  among the evaluated threshold candidates.

- Model training, threshold selection and conformal
  calibration use group-disjoint development subsets.

- Conformal quantiles are estimated from the
  development calibration subset.

- Nominal conformal coverage is 90%.

- Group-disjoint splitting alone does not establish
  a formal coverage guarantee for unseen machines.

- Validation status: DEVELOPMENT ONLY.

- The independent test set was previously accessed
  during model development and is therefore not
  a pristine final holdout.

- Test-set reference cases are used exclusively
  for software regression testing after model,
  threshold and conformal quantiles are fixed.

- Regression tests verify deployment consistency,
  not independent predictive accuracy or coverage.

- Random failures are excluded from model training,
  and their detection remains limited.

- Physical operating limits and out-of-distribution
  detection are not explicitly implemented.

- The external uncertainty review mechanism must
  be implemented at the multi-agent system level.

- The model is not validated for production deployment.

