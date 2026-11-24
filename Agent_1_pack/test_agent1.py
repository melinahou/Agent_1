
import json
import sys
import math
from pathlib import Path

from predict import predict_failure

REFERENCE_PATH = (
    Path(__file__).resolve().parent / "reference_cases.json"
)

def run_regression_test():

    with open(REFERENCE_PATH, "r") as f:
        cases = json.load(f)

    if len(cases) != 100:
        print("FAIL: Expected 100 reference cases")
        sys.exit(1)

    for case in cases:

        case_id = case["case_id"]

        machine_data = {
            key: float("nan") if value is None else value
            for key, value in case["machine_data"].items()
        }

        result = predict_failure(machine_data)

        # 1. Probability consistency
        if not math.isclose(
            result["failure_probability"],
            case["expected_probability"],
            abs_tol=1e-6,
            rel_tol=0
        ):
            print(f"FAIL: Case {case_id} — Probability mismatch")
            print("Expected:", case["expected_probability"])
            print("Actual:", result["failure_probability"])
            sys.exit(1)

        # 2. Decision consistency
        if result["prediction"] != case["expected_prediction"]:
            print(f"FAIL: Case {case_id} — Decision mismatch")
            print("Expected:", case["expected_prediction"])
            print("Actual:", result["prediction"])
            sys.exit(1)

        # 3. Conformal prediction set consistency
        if result["prediction_set"] != case["expected_prediction_set"]:
            print(f"FAIL: Case {case_id} — Prediction set mismatch")
            print("Expected:", case["expected_prediction_set"])
            print("Actual:", result["prediction_set"])
            sys.exit(1)

        # 4. Additional output consistency
        additional_fields = [
            "uncertainty_status",
            "send_to_agent2",
            "threshold",
            "conformal_alpha"
        ]

        for field in additional_fields:

            expected = case[f"expected_{field}"]
            actual = result[field]

            if actual != expected:
                print(f"FAIL: Case {case_id} — {field} mismatch")
                print("Expected:", expected)
                print("Actual:", actual)
                sys.exit(1)
  

    print(f"PASS: All {len(cases)} reference cases verified.")
    sys.exit(0)


if __name__ == "__main__":
    run_regression_test()
