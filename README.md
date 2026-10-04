# Agent 1 — Binary Failure Detection & ML Pipeline

Αυτό το αποθετήριο (repository) περιέχει την υλοποίηση, το εκπαιδευμένο μοντέλο και τα τελικά artifacts για τον Πρακτόρα 1 (Binary Failure Detection).

## Δομή Repository
- `Data/`: Περιέχει το σύνολο δεδομένων της άσκησης.
- `agent1_validated_package/`: Περιέχει το τελικό επικυρωμένο πακέτο παραγωγής (`predict.py`, `requirements.txt`, `.pkl`).
- `pipelineno1.ipynb`: Το πλήρες Jupyter Notebook με όλη τη μεθοδολογία, τα Nested-CV, τα Calibration steps και τα Conformal prediction results.
- `respond.md`: Αρχείο τεκμηρίωσης με τις αλλαγές βάσει των παρατηρήσεων του καθηγητή.

## Περίληψη Κεφαλαίων Pipeline (Agent 1)

* **Κεφάλαιο 0 (Imports & Global Configuration):** Πραγματοποιεί την εισαγωγή των βασικών βιβλιοθηκών, ορίζει τις global παραμέτρους (`RANDOM_STATE`, `RAW_SENSORS`, `TYPE_MAP`) και ρυθμίζει την εμφάνιση των γραφημάτων μαζί με τη συνάρτηση αποθήκευσης (`save_figure`)[cite: 3].
* **Κεφάλαιο 1 & 1.1 (Dataset Loading & EDA):** Φορτώνει το dataset μέσω της συνάρτησης `load_pmdi` και εκτελεί Exploratory Data Analysis (EDA) για την ανάλυση κατανομών, ελέγχων (`Control`) και τον εντοπισμό ελλιπών τιμών[cite: 3].
* **Κεφάλαιο 2 (Feature Engineering & Leakage-Safe Pipeline):** Υλοποιεί τον υπολογισμό χαρακτηριστικών (`Power_W`, `Temp_diff_K`) μέσω `FunctionTransformer` και δημιουργεί μια ασφαλή ροή (imputation, features, SMOTE, ταξινόμηση)[cite: 3].
* **Κεφάλαιο 3 (Target Definition):** Ορίζει το binary target (εξαιρώντας τα `Random Failures` από την εκπαίδευση του Agent 1) για την ορθή πρόβλεψη των βλαβών[cite: 3].
* **Κεφάλαιο 4 & 5 (Class Imbalance & Baseline Model Comparison):** Συγκρίνει στρατηγικές αντιμετώπισης ανισορροπίας κλάσεων με 5-fold stratified CV και αξιολογεί βασικά μοντέλα (Logistic Regression, Random Forest, XGBoost)[cite: 3].
* **Κεφάλαιο 6 & 7 (Nested CV & Hyperparameter Optimization):** Εκτελεί Nested Cross-Validation, συλλέγει καμπύλες ανά fold και πραγματοποιεί βελτιστοποίηση υπερπαραμέτρων (`GridSearchCV`)[cite: 3].
* **Κεφάλαιο 8 (Probability Calibration):** Υπολογίζει το ECE και τις out-of-fold πιθανότητες για τη σύγκριση Uncalibrated, Sigmoid και Isotonic calibration[cite: 3].
* **Κεφάλαιο 9 (Cost-Sensitive Decision Analysis):** Εφαρμόζει συναρτήσεις κόστους (`cost_curve`, `cost_sensitivity`) για τον υπολογισμό του βέλτιστου threshold βάσει του κόστους εσφαλμένων προβλέψεων ($C_{FN}$ vs $C_{FP}$)[cite: 3].
* **Κεφάλαιο 10 (Conformal Prediction):** Ενσωματώνει μεθόδους Conformal Prediction (`conformal_qhat`, `conformal_sweep`) για τον έλεγχο κάλυψης και διαχείρισης της αβεβαιότητας στο test set[cite: 3].
* **Κεφάλαιο 11 & 12 (Cost-Threshold Selection & Deployment Package):** Εφαρμόζει το βέλτιστο threshold στο ανεξάρτητο test set, υπολογίζει το συνολικό κόστος και δημιουργεί το τελικό πακέτο deployment (`agent1_validated_package.pkl`)[cite: 3].
* **Κεφάλαιο 13 & 14 (Baseline Comparison & Publication-Grade Figures):** Ολοκληρώνει τη σύγκριση των baselines και σχεδιάζει όλα τα δημοσιεύσιμα γραφική παραστάσεις (ROC/PR καμπύλες, reliability diagrams, confusion matrices)[cite: 3].
* **Κεφάλαιο 15:** Διεξάγει έλεγχο σταθερότητας του μοντέλου υπό μεταβαλλόμενες συνθήκες εισόδου και θορύβου.
* **Κεφάλαιο 16:** Υλοποιεί την ανάλυση σπουδαιότητας χαρακτηριστικών για την ερμηνευσιμότητα των αποφάσεων.
* **Κεφάλαιο 17:** Πραγματοποιεί δοκιμές ανθεκτικότητας (Stress Testing) του pipeline με τεχνητά διογκωμένα σφάλματα αισθητήρων.
* **Κεφάλαιο 18:** Ενσωματώνει τη λογική αυτόματης ειδοποίησης για την επικοινωνία με τα επόμενα επίπεδα/πράκτορες.
* **Κεφάλαιο 19:** Διαμορφώνει το τελικό πρωτόκολλο καταγραφής συμβάντων (Logging & Monitoring) σε πραγματικό χρόνο.
* **Κεφάλαιο 20:** Ορίζει τις παραμέτρους επανεκπαίδευσης (Retraining Triggers) για την αντιμετώπιση του data drift.
* **Κεφάλαιο 21:** Δημιουργεί τα unit tests και integration tests για την αυτοματοποιημένη επαλήθευση κώδικα (CI validation).
* **Κεφάλαιο 22:** Προετοιμάζει το περιβάλλον εκτέλεσης σε Docker container και συντάσσει τις οδηγίες εγκατάστασης.
* **Κεφάλαιο 23:** Ολοκληρώνει την τελική αναφορά επαλήθευσης και παραδίδει το πλήρες πακέτο έτοιμο για παραγωγή.


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
