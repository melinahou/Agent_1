# Agent 1 — Binary Failure Detection & ML Pipeline

Αυτό το αποθετήριο (repository) περιέχει την υλοποίηση, το εκπαιδευμένο μοντέλο και τα τελικά artifacts για τον Πρακτόρα 1 (Binary Failure Detection).

## Δομή Repository
- `Data/`: Περιέχει το σύνολο δεδομένων της άσκησης.
- `agent1_validated_package/`: Περιέχει το τελικό επικυρωμένο πακέτο παραγωγής (`predict.py`, `requirements.txt`, `.pkl`).
- `pipelineno1.ipynb`: Το πλήρες Jupyter Notebook με όλη τη μεθοδολογία, τα Nested-CV, τα Calibration steps και τα Conformal prediction results.
- `response.md`: Αρχείο τεκμηρίωσης με τις αλλαγές βάσει των παρατηρήσεων του καθηγητή.

## Μετρικά Απόδοσης Τελικού Μοντέλου (XGBoost)
Τα τελικά αποτελέσματα αξιολόγησης του μοντέλου στο ανεξάρτητο test set είναι:
- **Precision**: 0.7792
- **Recall**: 0.9091
- **F1-score**: 0.8392
- **ROC-AUC**: 0.995

## Βέλτιστο Threshold & Κατώφλι Απόφασης
- **Cost-optimal Threshold**: 0.15 (βελτιστοποιημένο με βάση το κόστος των ψευδώς αρνητικών και θετικών προβλέψεων).
- **Conformal Prediction (alpha=0.1)**: qhat = 0.0071, με empirical coverage 90.49%.

## Οδηγίες Χρήσης (Quickstart)
Για να τρέξετε τον κώδικα παραγωγής:
1. Εγκαταστήστε τις απαιτούμενες βιβλιοθήκες: `pip install -r agent1_validated_package/requirements.txt`
2. Εκτελέστε το script πρόβλεψης: `python agent1_validated_package/predict.py`
