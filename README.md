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

* **`0. Imports & Global Config`** – Εισάγει τις απαραίτητες βιβλιοθήκες και ορίζει τις παγκόσμιες ρυθμίσεις και το στυλ οπτικοποίησης.
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
* **`8. Probability Calibration`** – Συγκρίνει πιθανότητες με μεθόδους`Uncalibrated`, `Sigmoid` και `Isotonic` βάσει Brier score και ECE.
  * **`8.1 Fitted Calibration Models & Validation Curves`** – Εφαρμόζει τα μοντέλα καλιμπραρίσματος και παράγει τις αντίστοιχες καμπύλες επικύρωσης/αξιοπιστίας. Επιλέγουμε την `Isotonic`.
* **`9. Cost-Sensitive Decision Analysis & Optimal Threshold Selection`** – Υπολογίζει το βέλτιστο όριο απόφασης `t*` με βάση το οικονομικό κόστος των C_FN=10 και C_FP=1.
  * **`9.1 Cost Matrix Definition & Expected Loss Minimization`** – Αξιολογεί το t* και κάνει ανάλυση ευσαισθησίας.
  * **`9.2 Sensitivity Analysis on Cost Ratios`** – Αναλύει τη μεταβολή του βέλτιστου ορίου συναρτήσει διαφορετικών λόγων κόστους (Cost Ratios).
  * **`9.3 Threshold Bootstrap Confidence Intervals`** – Υπολογίζει διαστήματα εμπιστοσύνης (Bootstrap CIs) για το βέλτιστο όριο απόφασης.
* **`10. Mondrian Conformal Prediction Functions`** – Υλοποιεί προβλέψεις συμμόρφωσης ανά κλάση (ξεχωριστά q_hat) για έλεγχο κάλυψης και αβεβαιότητας.
  * **`10.1 Execution of Mondrian Conformal Uncertainty & Coverage Sweep`** – Εκτυπώνει αποτελέσματα για διάφορα alpha (0.010 , 0.3).
  * **`10.2 Proper Mondrian Split-Conformal Evaluation Protocol`** – Χωρίζει τα δεδομένα σε 3 σετ: `training`, `conformal calibration` και `independent test`. Εκπαιδεύει calibrated weighted XGBoost μόνο στο training set. Υπολογίζει ξεχωριστά q_hat ανά κλάση μόνο στο conformal calibration set. Ελέγχει κάλυψη (συνολική και ανά κλάση) και prediction-set behavior μόνο στο independent test set.
  * **`10.3 Execution and Storage of Mondrian Split-Conformal Evaluation Results`** – Εκτελεί, εκτυπώνει και αποθηκεύει τα αποτελέσματα.
* **`11. Independent Test Evaluation`** – Εφαρμόζει πρωτόκολλο split-conformal και αξιολογεί το μοντέλο σε ανεξάρτητο σύνολο δοκιμών.
* **`12. Publication-Grade Figure Definitions & Styling Functions`** – Περιέχει συναρτήσεις οπτικοποίησης για τη δημιουργία επαγγελματικών γραφημάτων υψηλής ανάλυσης.
  * **`12.1 Execute and Render All Publication-Grade Figures`** – Εκτελεί τη μαζική παραγωγή, αποθήκευση και προβολή όλων των γραφημάτων σε μορφές PNG, PDF και SVG.
* ** `13. Save Validated Agent 1 Deployment Package`** - Αποθηκεύει σε pkl αρχείο το εκπαιδευμένο βαθμονομημένο και δοκιμασμένο μοντέλο, τα χαρακτηριστικά και δεδομένα ( raw features , type map), τις παραμέτρους αβεβαιότητας και κόστους(t*=0.05, C_FN=10, C_FP=1, qhats, alpha=0.10).
* **`14.Agent 1 Inference Class Definition`**- Διαβάζει το αρχείο pkl, παίρνει τα ωμά δεδομένα και εκτυπώνει την καλιμπραρισμένη πιθανότητα βάση το threshold, εκτυπώνει την δυαδική απόφαση και εφαρμόζει Mondrian Conformal Prediction για να μας πει πόσο σίγουρο είναι.
* **`14.1 Execute Package Load Validation Test`**- Επιβεβαίωση φόρτωσης μέσω της κλάσης.
* **`15. RNF Stress Test`**- Εκτελεί έναν έλεγχο αντοχής (stress test) τροφοδοτώντας το μοντέλο με τα δείγματα τυχαίων σφαλμάτων (Random Failures) που είχαν εξαιρεθεί από την εκπαίδευση, και υπολογίζει στατιστικά απόδοσης, ποσοστά ανίχνευσης και δείκτες αβεβαιότητας για αυτά..
* **`16. Final Validated Results Summary Table`**- Δημιουργεί έναν τελικό, συγκεντρωτικό πίνακα αναφοράς (summary table) που περιλαμβάνει όλες τις βασικές μετρήσεις απόδοσης του μοντέλου από το Nested-CV, τα αποτελέσματα στο ανεξάρτητο test set, τις παραμέτρους κόστους και τα ποσοστά κάλυψης του conformal prediction.
* **`17. Main Execution Workflow Block for Agent 1 Pipeline`**- Εκτελεί την κεντρική ροή εργασίας (main workflow) ολόκληρου του pipeline για τον Agent 1, εκτελώντας διαδοχικά την εκπαίδευση, τη βελτιστοποίηση υπερπαραμέτρων, τη βαθμονόμηση, την ανάλυση κόστους, το split-conformal validation και αποθηκεύοντας τελικά το έτοιμο πακέτο μοντέλου.
* **`18. Write Standalone Inference Script (predict.py) for Production Deployment`**- Δημιουργεί και αποθηκεύει ένα αυτόνομο script παραγωγής (predict.py) το οποίο ενσωματώνει όλη τη λογική φόρτωσης του μοντέλου, τους μετασχηματιστές και την κλάση inference για την αξιολόγηση νέων δεδομένων και τη διασύνδεση με τον Agent 2.
    * **`18.1 Import and Test Prediction Module`**- Δημιουργεί και αποθηκεύει ένα script ελέγχου λειτουργίας (test_agent1.py) το οποίο εισάγει τη συνάρτηση πρόβλεψης και πραγματοποιεί έναν έλεγχο  με δοκιμαστικά δεδομένα μηχανής.
    * **`18.2 Write Package Dependencies (requirements.txt) for Deployment`**- Δημιουργεί και αποθηκεύει το αρχείο εξαρτήσεων requirements.txt στο πακέτο παραγωγής, καταγράφοντας όλες τις απαραίτητες βιβλιοθήκες Python που απαιτούνται για τη λειτουργία του μοντέλου.
    * **`18.3 Compress Deployment Package & Trigger Download (.zip)`**-Συμπιέζει ολόκληρο τον φάκελο του πακέτου παραγωγής (agent1_package) σε αρχείο μορφής .zip και ενεργοποιεί την αυτόματη λήψη του (download) στον υπολογιστή μέσω του Google Colab. 

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
