
import sys
import os

# Ασφαλής εύρεση διαδρομής (συμβατή και με Colab και με Terminal)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = '/content/agent1_package'

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from predict import predict_failure

def run_smoke_test():
    print("=" * 50)
    print("Εκτέλεση Δοκιμής Agent 1 (Smoke Test)...")
    print("=" * 50)
    
    sample_data = {
        'Type': 'M',
        'Air_temperature_K': 300.5,
        'Process_temperature_K': 311.2,
        'Rotational_speed_rpm': 1350,
        'Torque_Nm': 65.0,
        'Tool_wear_min': 200
    }
    
    try:
        result = predict_failure(sample_data)
        print("\n[ΕΠΙΤΥΧΙΑ] Η πρόβλεψη εκτελέστηκε κανονικά!")
        print("-" * 40)
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("-" * 40)
    except Exception as e:
        print(f"\n[ΣΦΑΛΜΑ] Απέτυχε η πρόβλεψη: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_smoke_test()
