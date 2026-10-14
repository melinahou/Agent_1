# Agent 1: Predictive Maintenance Failure Detection System / Σύστημα Ανίχνευσης Βλαβών Προληπτικής Συντήρησης

## 📌 Overview / Επισκόπηση
This repository contains the production-ready machine learning inference package for **Agent 1**, designed for predictive maintenance in industrial machinery (based on the AI4I dataset). 

Το σύστημα αξιοποιεί ένα βελτιστοποιημένο μοντέλο δυαδικής ταξινόμησης **XGBoost**, ενσωματωμένο με custom feature engineering, καλιμπράρισμα πιθανοτήτων και **Split-Conformal Prediction**, ώστε να εντοπίζει αξιόπιστα μηχανικές βλάβες (όπως φθορά εργαλείου, υπερφορτώσεις ισχύος και αστοχίες απαγωγής θερμότητας), παρέχοντας παράλληλα στατιστικά εγγυημένα όρια αβεβαιότητας για κάθε πρόβλεψη.

---

## 📦 What's Inside the ZIP Package / Περιεχόμενα Πακέτου
Το αρχείο συμπίεσης (`agent1_package.zip`) περιλαμβάνει όλα τα απαραίτητα στοιχεία για την αυτόνομη εκτέλεση του συμπερασμού:

1. **`predict.py`**  
   - Η βασική μονάδα (module) παραγωγής που εκθέτει τη δημόσια συνάρτηση `predict_failure(machine_data)`. Διαχειρίζεται την αυτόματη ευθυγράμμιση των χαρακτηριστικών εισόδου, την εφαρμογή των καθορισμένων ορίων απόφασης και την αποσυμπίεση/φόρτωση του μοντέλου.
   
2. **`test_agent1.py`**  
   - Ένα αυτόνομο σενάριο δοκιμής (smoke test) γραμμής εντολών. Φορτώνει το πακέτο, τροφοδοτεί το σύστημα με ένα αντιπροσωπευτικό δείγμα βιομηχανικών δεδομένων αισθητήρων και επαληθεύει την ορθή εκτέλεση από άκρη σε άκρη.

3. **`agent1_validated_package.pkl`**  
   - Το αποθηκευμένο (serialized) αρχείο παραγωγής που περιέχει τον εκπαιδευμένο εκτιμητή XGBoost, τους προσαρμοσμένους μετασχηματιστές προεπεξεργασίας, τα βέλτιστα κατώφλια απόφασης και τα μεταδεδομένα conformal prediction (`q_hat`).

4. **`requirements.txt`**  
   - Ο ακριβής κατάλογος των απαιτούμενων βιβλιοθηκών (όπως `xgboost`, `scikit-learn`, `numpy`, `pandas`) που χρειάζονται για τη λειτουργία του περιβάλλοντος.

---

## 🚀 How to Use & Test / Τρόποι Εκτέλεσης και Δοκιμής

Μπορείτε να δοκιμάσετε και να αξιολογήσετε το μοντέλο χρησιμοποιώντας έναν από τους παρακάτω τρόπους:

### 1. Google Colab Execution / Εκτέλεση σε Google Colab
* **Μεταφόρτωση και Αποσυμπίεση:** Ανεβάστε το συμπιεσμένο πακέτο (`agent1_package.zip`) σε μια νέα καρτέλα στο Google Colab και εκτελέστε τη διαδικασία εξαγωγής των αρχείων του στον επιθυμητό φάκελο.
* **Φόρτωση Διεπαφής (Πρώτο Κελί):** Ανοίξτε ή εισαγάγετε το βασικό αρχείο συμπερασμού (`predict.py`) για να ενεργοποιήσετε τη βασική συνάρτηση πρόβλεψης και να προετοιμάσετε το περιβάλλον.
* **Εκτέλεση Δοκιμής (Δεύτερο Κελί):** Σε ένα αμέσως επόμενο κελί, εκτελέστε το έτοιμο σενάριο δοκιμής (`test_agent1.py`) τροφοδοτώντας το μοντέλο με ένα δείγμα βιομηχανικών παραμέτρων, ώστε να λάβετε άμεσα τα αποτελέσματα πρόβλεψης και αβεβαιότητας.

### 2. Standalone Terminal Execution (CLI) / Αυτόνομη Εκτέλεση από το Τερματικό
* **Προετοιμασία Περιβάλλοντος:** Αποσυμπιέστε τον φάκελο τοπικά στον υπολογιστή σας και εγκαταστήστε τις απαραίτητες βιβλιοθήκες και εξαρτήσεις από το αρχείο `requirements.txt`.
* **Εκτέλεση Σεναρίου Δοκιμής:** Τρέξτε απευθείας το αρχείο επαλήθευσης (`test_agent1.py`) από τη γραμμή εντολών, επιτρέποντας στο σύστημα να φορτώσει αυτόματα το μοντέλο και να εμφανίσει τα αποτελέσματα στην οθόνη.

---
## 📂 Repository Structure / Δομή Αποθετηρίου

* **`Agent1_pack/`** - Ο φάκελος που περιέχει τα βασικά αρχεία παραγωγής του Agent 1 (`predict.py`, `test_agent1.py`, το `.pkl` και το `requirements.txt`).
* **`Data/`** - Ο φάκελος με τα δεδομένα (dataset) που χρησιμοποιήθηκαν στο pipeline του Agent 1.
* **`pipeline/`** - Ο φάκελος που φιλοξενεί την υλοποίηση του pipeline.
* **`README.md`** - Η κεντρική τεκμηρίωση και οδηγός χρήσης του έργου.
* **`respond`** - Το αρχείο με τις απαντήσεις και την τεκμηρίωση προς τον καθηγητή. 
---

## ⚙️ Code Structure / Δομή Κώδικα

* **`0. Imports & Global Config`** – Εισάγει τις απαραίτητες βιβλιοθήκες (`pandas`, `numpy`, `scikit-learn`, `xgboost`, κ.λπ.) και ορίζει τις παγκόσμιες ρυθμίσεις και το στυλ οπτικοποίησης.
* **`1. Dataset Loading & Structural Preprocessing`** – Φορτώνει το αρχείο CSV, μετατρέπει τις ημερομηνίες σε datetime, ταξινομεί τα δεδομένα.
  * **`1.1 Exploratory Data Analysis (EDA)`** – Ελέγχει την κατανομή των βλαβών (`Diagnostic`), τους ελέγχους (`Control`) και τον εντοπισμό ελλιπών τιμών.
* **`2. Feature Engineering`** – Δημιουργεί έναν custom μετασχηματιστή (`FeatureEngineer`) για τον υπολογισμό νέων χαρακτηριστικών όπως η διαφορά θερμοκρασίας και το γινόμενο ροπής επί φθοράς.
  * **`2.1 Leakage-Safe Pipeline Construction`** – Κατασκευάζει ασφαλές `ImbPipeline` που ενσωματώνει binary δείκτες απουσίας, KNN imputer και τεχνικές εξισορρόπησης.
* **`3. Target Definition`** – Ορίζει το binary target για τον Agent 1 (0 = No Failure, 1 = Predictable Failure) εξαιρώντας τα τυχαία σφάλματα.
* **`4. Class Imbalance Strategy Comparison`** – Συγκρίνει στρατηγικές διαχείρισης ανισορροπίας κλάσεων χρησιμοποιώντας `GroupKFold` ανά σύστημα.
* **`5. Baseline Model Comparison`** – Συγκρίνει βασικούς αλγορίθμους (Logistic Regression, Random Forest, XGBoost) ως προς την απόδοσή τους.
* **`6. Rigorous Nested Cross-Validation & Hyperparameter Optimization`** – Εκτελεί εξωτερικό και εσωτερικό βρόχο επικύρωσης με `GroupKFold` και `GridSearchCV` για βέλτιστη ρύθμιση υπερπαραμέτρων.
  * **`6.1 Nested CV Execution & Performance Aggregation`** – Εκτελεί τον πλήρη βρόχο Nested CV ανά fold και συγκεντρώνει τα μετρικά αποτελέσματα απόδοσης.
* **`7. Final Hyperparameter Optimization`** – Εφαρμόζει τελικό Grid Search στο σύνολο των δεδομένων και συλλέγει δεδομένα καμπυλών ανά fold.
* **`8. Probability Calibration`** – Συγκρίνει πιθανότητες με μεθόδους `Sigmoid` και `Isotonic` βάσει Brier score και ECE.
  * **`8.1 Fitted Calibration Models & Validation Curves`** – Εφαρμόζει τα μοντέλα καλιμπραρίσματος και παράγει τις αντίστοιχες καμπύλες επικύρωσης/αξιοπιστίας.
* **`9. Cost-Sensitive Decision Analysis & Optimal Threshold Selection`** – Υπολογίζει το βέλτιστο όριο απόφασης ($t^*$) με βάση το οικονομικό κόστος των $C_{FN}$ και $C_{FP}$.
  * **`9.1 Cost Matrix Definition & Expected Loss Minimization`** – Ορίζει τον πίνακα κόστους και ελαχιστοποιεί την αναμενόμενη ζημία για τον καθορισμό του ορίου.
  * **`9.2 Sensitivity Analysis on Cost Ratios`** – Αναλύει τη μεταβολή του βέλτιστου ορίου συναρτήσει διαφορετικών λόγων κόστους (Cost Ratios).
  * **`9.3 Threshold Bootstrap Confidence Intervals`** – Υπολογίζει διαστήματα εμπιστοσύνης (Bootstrap CIs) για το βέλτιστο όριο απόφασης.
* **`10. Mondrian Conformal Prediction Functions`** – Υλοποιεί προβλέψεις συμμόρφωσης ανά κλάση για έλεγχο κάλυψης και αβεβαιότητας.
  * **`10.1 Nonconformity Measure Implementation`** – Ορίζει τη συνάρτηση μη-συμμόρφωσης βάσει των προβλεπόμενων πιθανοτήτων.
  * **`10.2 Calibration Set P-Value Computation`** – Υπολογίζει τις τιμές p-values για το σύνολο βαθμονόμησης/συμμόρφωσης.
  * **`10.3 Prediction Set Generation & Efficiency Metrics`** – Παράγει τα σύνολα πρόβλεψης (prediction sets) και υπολογίζει τους δείκτες αποδοτικότητας και κάλυψης.
* **`11. Independent Test Evaluation`** – Εφαρμόζει πρωτόκολλο split-conformal και αξιολογεί το μοντέλο σε ανεξάρτητο σύνολο δοκιμών.
* **`12. Publication-Grade Figure Definitions & Styling Functions`** – Περιέχει συναρτήσεις οπτικοποίησης για τη δημιουργία επαγγελματικών γραφημάτων υψηλής ανάλυσης.
  * **`12.1 Execute and Render All Publication-Grade Figures`** – Εκτελεί τη μαζική παραγωγή, αποθήκευση και προβολή όλων των γραφημάτων σε μορφές PNG, PDF και SVG.
* ** `13. Advanced Data Persistence & Artifact Serialization Pipeline`** - Ορίζει μηχανισμούς και συναρτήσεις για την ασφαλή αποθήκευση, συμπίεση και διαχείριση όλων των εκπαιδευμένων μοντέλων, των transformer pipelines, των καμπυλών και των serialised αντικειμένων σε αρχεία (π.χ. με joblib και pickle).
  * **`13.1 Execute Full Artifact Serialization`**-Εκτελεί την οριστική αποθήκευση όλων των παραγόμενων αρχείων, μοντέλων και διαμερισμάτων δεδομένων στον τοπικό φάκελο εργασίας για μελλοντική επαναχρησιμοποίηση.
* **`14. Integrated Real-Time Inference Wrapper (Agent 1 Prediction Engine)`**- Δημιουργεί μια ενοποιημένη κλάση/συνάρτηση wrapper (Agent1InferenceEngine) η οποία αναλαμβάνει την αυτόματη επεξεργασία νέων δεδομένων εισόδου, εφαρμόζει τους μετασχηματισμούς, υπολογίζει τις βαθμονομημένες πιθανότητες και παρέχει την τελική απόφαση βάσει του βέλτιστου ορίου κόστους ($t^*$).
* **`14.1 Run Inference Engine on Sample Holdout Data`**- Δοκιμάζει το σύστημα inference σε δείγματα από το holdout/test set, επιβεβαιώνοντας την ορθότητα παραγωγής προβλέψεων και εκτιμήσεων αβεβαιότητας σε πραγματικό χρόνο.
* **`15. Comprehensive Model Explainability & Feature Importance (SHAP Analysis)`**- Εφαρμογή ανάλυσης SHAP (SHapley Additive exPlanations) για το τελικό εκπαιδευμένο μοντέλο XGBoost, εξάγοντας τόσο τις παγκόσμιες (global) όσο και τις τοπικές (local) τιμές συνεισφοράς των χαρακτηριστικών.
* **`15.1 Generate and Save SHAP Summary & Dependence Plots`**- Παράγει και αποθηκεύει τα αντίστοιχα διαγράμματα SHAP (summary plots, bar plots και dependence plots) για την ερμηνευσιμότητα των αποφάσεων του μοντέλου πρόβλεψης αστοχίας.
* **`16. Error Analysis & Edge Case Misclassification Diagnostic Suite`**- Δημιουργεί μια σουίτα διαγνωστικού ελέγχου για τον εντοπισμό και την ανάλυση των περιπτώσεων όπου το μοντέλο απέτυχε (False Positives και False Negatives), εξετάζοντας τα κοινά χαρακτηριστικά των λαθών.
* **`17. Automated Documentation & Markdown Report Generator`**- Συνθέτει αυτόματα μια περιεκτική αναφορά σε μορφή Markdown (Agent1_Technical_Report.md) η οποία περιλαμβάνει συγκεντρωτικούς πίνακες μετρικών, υπερπαραμέτρους, αποτελέσματα conformal prediction και οδηγίες χρήσης.
* **`18. Final Execution Orchestration, Logging, and Sanity Checks`**- Εκτελεί συνολικό έλεγχο ορθότητας (sanity checks) σε όλα τα ενδιάμεσα βήματα, διασφαλίζοντας την ακεραιότητα των δεδομένων και την απρόσκοπτη λειτουργία του κώδικα από την αρχή μέχρι το τέλος.
* **`18.1 End-to-End Pipeline Execution Trigger`**- Εκκινεί την αυτοματοποιημένη, διαδοχική εκτέλεση ολόκληρου του pipeline από το αρχικό CSV έως την τελική αναφορά και τα γραφήματα.
* **`18.2 System Validation & Pipeline Health Report`**- Ελέγχει τα logs εκτέλεσης, επαληθεύει ότι όλα τα αρχεία εξόδου (models, figures, metrics tables) έχουν δημιουργηθεί επιτυχώς και εκτυπώνει την τελική αναφορά κατάστασης (health report).
* **`18.3 Pipeline Termination & Deployment Readiness Confirmation`**- Επιβεβαιώνει την επιτυχή ολοκλήρωση της ροής εργασίας (pipeline termination), πιστοποιώντας ότι το σύστημα είναι πλήρως έτοιμο για παραγωγική λειτουργία (deployment readiness) και ενσωμάτωση στο αποθετήριο.

---


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
