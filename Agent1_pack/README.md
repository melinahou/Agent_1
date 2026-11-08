# Agent 1 — Predictive Maintenance

## Package contents

- agent1_validated_package.pkl: Trained and calibrated model
- agent1_components.py: Preprocessing components
- predict.py: Prediction interface
- test_agent1.py: Smoke test
- requirements.txt: Python dependencies

## 1. Setup in Google Colab

Extract the ZIP and upload all six files directly to /content/ in Google Colab.

Install dependencies:

    %pip install -r /content/requirements.txt

Restart the runtime if required.

## 2. Run the smoke test

    !python /content/test_agent1.py

## 3. Run a prediction

Create a new Colab cell and paste:

    import sys
    sys.path.insert(0, "/content")

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
    print(result)

Expected failure probability for this sample:
0.992593 (approximately, with compatible dependencies).

## Notes

The prediction function returns:
- Calibrated failure probability
- Cost-based failure decision
- Conformal prediction set and uncertainty state
- Optional payload for Agent 2

Do not copy the full contents of predict.py into a Colab cell.
Import the module as shown above.
