# Agent 1 Pipeline Refactoring & Architecture Documentation

Σύμφωνα με τις παρατηρήσεις και τις αξιολογήσεις, το pipeline του **Agent 1** αναδομήθηκε πλήρως. 
Παρακάτω παρατίθενται αναλυτικά οι διορθώσεις, οι αρχιτεκτονικές αλλαγές και η αντιστοίχισή τους 
ανά τμήματα του Jupyter Notebook .

---


## 1. Random Failures (RNF) & Tool Wear Handling

### Original Issue
The AI4I dataset includes Random Failures (RNF), which do not follow the same predictable patterns as other machine failure mechanisms.

Additionally, tool wear is a relevant predictor of machine failure, but its relationship with failure should be modeled without introducing unnecessary feature-specific hyperparameter optimization.

### Implemented Changes

**1. Exclusion of Random Failures from Model Training**

- Random Failures are excluded from the binary classification training target.
- Agent 1 focuses on predicting failure mechanisms considered predictable from the available sensor measurements.
- This distinction is implemented in **Chapter 3 — Target Definition and Failure Label Preparation**.

**2. Random Failure Robustness Assessment**

- Excluded Random Failure observations are subsequently used for a separate robustness assessment.
- The objective is to examine how the trained model behaves when presented with failure events outside its primary training target.
- This analysis is implemented in **Chapter 13.2 — Robustness Assessment Using Excluded Random Failures**.
- The results are interpreted as a stress test rather than as evidence of reliable Random Failure prediction.

**3. Tool Wear Feature Handling**

- Tool wear is retained as an input feature (`Tool_wear_min`).
- Its interaction with torque is represented through the engineered feature `Torque_x_Wear`.
- Feature engineering is integrated into the preprocessing pipeline to ensure consistent transformations during training and inference.
- The relevant implementation is documented in **Chapter 2 — Feature Engineering and Leakage-Safe Pipeline**.

### Outcome

Agent 1 is trained to identify predictable machine failures using the available operating measurements, while Random Failures are evaluated separately through a robustness assessment.

Tool wear remains part of the model's input feature set, including its interaction with torque, without requiring a separate tool-wear-specific prediction model.


---

## 2. Αποφυγή Data Leakage (Interpolation εντός Nested-CV) 
* **Σημεία στον κώδικα:** 
    1)  Δημιουργία Leakage-Safe Pipeline (Τμήμα 2.1 - Leakage-Safe Pipeline Construction):
Για να μην υπάρχει καμία «κλεφτή ματιά» στα δεδομένα ελέγχου, μεταφέραμε όλη τη διαδικασία 
προεπεξεργασίας (όπως τη δημιουργία binary δεικτών απουσίας MissingIndicatorAdder και τον KNNImputer) 
μέσα σε ένα ενιαίο, κλειστό pipeline της βιβλιοθήκης imblearn (ImbPipeline).
    2) Ενσωμέτρηση μέσα στο Nested Cross-Validation (Τμήματα 4, 6 & 17): Με αυτόν τον τρόπο, 
όταν το μοντέλο χωρίζεται σε folds για το Nested Cross-Validation ή το Grid Search, ο imputer και 
οι μετασχηματιστές εφαρμόζονται αποκλειστικά και μόνο πάνω στο training fold κάθε φορά και ποτέ 
στο validation/test fold. Έτσι, το μοντέλο μαθαίνει τις στατιστικές τιμές (όπως τους μέσους όρους 
ή τους πλησιέστερους γείτονες) μόνο από τα δεδομένα εκπαίδευσης, μηδενίζοντας πλήρως το data 
leakage.

---

## 3. Σειρά Εκτέλεσης Features & Validation Artifacts 
* **Σημεία στον κώδικα:** 
1) Καθολικός Ορισμός Χαρακτηριστικών στην Αρχή (Τμήματα 2 & 3): 
Μεταφέραμε όλη τη μηχανική χαρακτηριστικών (feature engineering), τους δείκτες απουσίας και τυχόν 
αλληλεπιδράσεις (όπως τα Temp_diff_K και τα σχετικά features) στην αρχή του κώδικα, πριν καν 
ξεκινήσει οποιαδήποτε διαδικασία Nested Cross-Validation, GridSearchCV ή calibration.
2) Ενιαίο και Συνεπές Dataset (Τμήμα 17 - Main Execution Workflow):
Διασφαλίσαμε ότι το τελικό Pipeline εκπαιδεύεται, συντονίζεται (tuned), βαθμονομείται (calibrated) 
και αποθηκεύεται (agent1_validated_package.pkl) χρησιμοποιώντας ακριβώς το ίδιο, πλήρες και σταθερό 
σύνολο χαρακτηριστικών από την πρώτη μέχρι την τελευταία γραμμή, εξαλείφοντας πλήρως την απόκλιση 
ανάμεσα στα ενδιάμεσα νούμερα και το τελικό αποθηκευμένο artifact.
---

## 4. Βελτιστοποίηση Threshold βάσει Κόστους (Cost-Optimal Threshold) 
* **Σημεία στον Κώδικα:**
1)  Εισαγωγή Ανάλυσης Κόστους & Λόγου C_FN/C_FP=10:1 (Τμήμα 9 και 11 ):
Αντί να αφήσουμε το κατώφλι στο αυθαίρετο 0.5, ορίσαμε ρητά λόγο κόστους C_FN/C_FP=10:1 (όπου μια 
απροειδοποίητη αστοχία κοστίζει πολλαπλάσια από έναν περιττό έλεγχο).
2) Υπολογισμός Βέλτιστου Ορίου t* μέσω Cost Curve (Τμήματα 9.1 , 9.2 , 9.3):
Προσθέσαμε τη συνάρτηση εύρεσης καμπύλης κόστους (cost_curve), η οποία σαρώνει όλες τις πιθανότητες 
και βρίσκει το ακριβές, βέλτιστο κατώφλι απόφασης (t*) που ελαχιστοποιεί το συνολικό οικονομικό 
κόστος. Όπως φαίνεται και στον τελικό πίνακα, το κατώφλι δεν είναι πλέον 0.5, αλλά t* = 0.049, 
προσαρμοσμένο ακριβώς στις απαιτήσεις της βιομηχανικής πραγματικότητας
---

## 5. Αντικατάσταση Απλού Certainty με Mondrian Conformal Prediction 
* **Σημεία στον Κώδικα:**
1) Υιοθέτηση Mondrian Conformal Prediction (Τμήματα 10, 10.1, 10.2, 10.3, 11 & 17): 
Αντί για πρόχειρα κριτήρια αβεβαιότητας, εφαρμόσαμε Mondrian Conformal Prediction (με ονομαστική 
κάλυψη alpha = 0.10, άρα 90% εγγυημένη μαθηματική κάλυψη).
2) Παραγωγή Prediction Sets & Uncertainty States (Τμήμα 17 & 18):
Το σύστημα πλέον δεν επιστρέφει απλώς μια «ναι/όχι» πρόβλεψη ή ένα ασαφές ποσοστό, αλλά ένα αυστηρό 
σύνολο πρόβλεψης (prediction_set) ανά κλάση μέσω των ποσοτηρίων qhat (qhat = 0.002 για την κλάση 0 
και 0.8758 για την κλάση 1). Έτσι, όταν ένα δείγμα πέφτει σε περιοχή αβεβαιότητας (π.χ. άδειο ή 
διφορούμενο σύνολο), χαρακτηρίζεται επισήμως ως uncertain με βάση τη στατιστική θεωρία των 
Conformal Predictors και όχι τυχαία thresholds.
---

## 6. Διαχείριση των ελλιπών δεδομένων (missing data) στο dataset AI4I PMDI
* **Σημεία στον Κώδικα:**
1) Προσθήκη Missing Indicators ως explicit features (Τμήματα 2, 2.1):
Για να διασφαλίσουμε τη διαφάνεια και να προλάβουμε την κριτική των reviewers, δεν αφήσαμε τον 
KNNImputer να συμπληρώνει τα κενά "αθόρυβα". Προσθέσαμε ρητά δυαδικούς δείκτες απουσίας 
(Air_temperature_K_isnan και Process_temperature_K_isnan) μέσα στα χαρακτηριστικά εισόδου του 
μοντέλου (X_new και FeatureEngineer). Έτσι, το μοντέλο μαθαίνει να ξεχωρίζει πότε μια θερμοκρασία 
είναι πραγματική μέτρηση και πότε είναι αποτέλεσμα imputation.
---

## 7. Ταξινόμηση 
* **Σημεία στον Κώδικα:**
1) Καθολική Χρήση του GroupKFold ανά Μηχανή / System (UID) (Τμήματα 4 και 7):
Αντί για τυχαίο ανακάτεμα, ορίσαμε ρητά τη στήλη των συστημάτων/μηχανών ως ομάδα 
(groups_col = X['UID']) και περάσαμε το GroupKFold σε όλες τις βασικές συναρτήσεις αξιολόγησης 
(nested_cv_evaluate και compare_models).Με αυτόν τον τρόπο, ολόκληρες μηχανές (systems) κρατούνται 
αυστηρά απομονωμένες: ό,τι ανήκει σε μια μηχανή πηγαίνει εξ ολοκλήρου είτε στο training fold είτε 
στο validation/test fold. Έτσι, το μοντέλο δοκιμάζεται σε εντελώς άγνωστες μηχανές, αποδεικνύοντας 
ότι τα νούμερα απόδοσης (όπως το υψηλό ROC-AUC και οι μετρικές generalization) είναι απολύτως 
αληθινά και ανθεκτικά.
---

## 8. Στατιστική εγκυρότητα (Bootstrap)
* **Σημεία στον Κώδικα:**
1) Εφαρμογή Bootstrap Confidence Intervals (CIs) (Τμήματα 9.3 και 11 ):
Ενσωματώσαμε αλγόριθμο Bootstrap resampling κατά την αξιολόγηση του ανεξάρτητου test set. Αντί για 
έναν σκέτο αριθμό, το σύστημα εκτελεί πολλαπλές επαναδειγματοληψίες (es. 500 iterations) για να 
υπολογίσει διαστήματα εμπιστοσύνης (π.χ. 95% CI) τόσο για το βέλτιστο όριο κόστους (t*) όσο και για 
το Precision και το Recall, δίνοντας ακριβώς τη «ζώνη ασφαλείας».
---

## 9. Weighting και Calibration σαν «ζώνη και τις τιράντες»
* **Σημεία στον Κώδικα:**
1) Αξιολόγηση Plain XGBoost vs Weighted (Τμήματα 4 και 5):
Συγκρίναμε την απόδοση του βασικού αλγορίθμου χωρίς τεχνητά βάρη (scale_pos_weight = 1) σε σχέση με 
τις σταθμισμένες εκδόσεις, επιβεβαιώνοντας ότι το plain XGBoost διατηρεί ανώτερη ικανότητα κατάταξης 
(με υψηλότερο Average Precision, π.χ. κοντά στο 0.932).
2) Διαχωρισμός Ranking και Cost-Sensitive Thresholding (Τμήματα 8, 8.1, 9, 9.1, 9.2):
Απαλλάξαμε το μοντέλο από το διπλό βάρος: αφήνουμε το plain XGBoost να παράγει άριστες, 
αδιάφθορες πιθανότητες κατάταξης, εφαρμόζουμε την Isotonic Calibration πάνω σε καθαρή βάση, και στη 
συνέχεια αναθέτουμε στον λόγο κόστους (C_FN/C_FP) και στο βέλτιστο κατώφλι (t*) την ευθύνη της 
τελικής απόφασης. Έτσι, το μοντέλο διατηρεί τη μέγιστη στατιστική του ισχύ.
---

## 10. Λεπτομερειες στην predict.py
* **Σημεία στον Κώδικα:**
1) Το πρόβλημα: Αν τυχόν υπήρχε κώδικας δοκιμής ή smoke test εκτός προστατευμένου μπλοκ, κάθε φορά 
που ο Agent 2 θα εισήγαγε (import) τον Agent 1, θα εκτελούνταν αυτόματα δοκιμαστικές προβλέψεις και 
εκτυπώσεις στην κονσόλα.
Η διόρθωση: Διασφαλίσαμε ότι οποιοδήποτε test ή demo σενάριο περικλείεται αυστηρά κάτω από το 
if __name__ == '__main__':. Έτσι, η εισαγωγή του predict.py γίνεται «σιωπηλά» και με ασφάλεια, 
έτοιμη να εξυπηρετήσει αιτήματα μόνο όταν το ζητήσει ο Agent 2.
2) Το πρόβλημα: Το __main__.engineer_features = ... είναι μπάλωμα.
Η διόρθωση: Δεν έγινε :) 











