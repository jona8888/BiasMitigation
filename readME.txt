# Bias Mitigation in k-Nearest Neighbors (kNN)

This project investigates bias mitigation in k-Nearest Neighbors (kNN) classification 
by applying fairness-aware preprocessing techniques. 
The goal is to reduce gender bias in decision-making systems without sacrificing model accuracy.

#Objective
To evaluate and improve fairness in kNN models using the following methods:
- Undersampling: Reducing samples from overrepresented groups.
- Oversampling (SMOTE)**: Increasing samples for underrepresented groups.
- Novel Feature Weighting**: Emphasizing socially impactful features (e.g., education, hours worked).

#Datasets
- Adult Census Income Dataset**: Predicts if income > $50K based on demographic/work features.
- German Credit Dataset**: Classifies individuals as good or bad credit risk based on financial and personal attributes.

#Methodology
1. Baseline – No mitigation (see `knn682.py`)
2. Fairness-Aware Sampling:
   - Undersampling: Drop majority group samples
   - Oversampling: SMOTE to synthesize minority class
3. Novel Technique**:
   - Feature weighting + undersampling for balanced fairness and accuracy

#Project Structure

| File                        | Description |
|-----------------------------|-------------|
| `knn682.py`                 | Baseline kNN** on the Adult dataset with no fairness adjustments. |
| `Adult1_UNDERsam.py`        | kNN on Adult dataset using undersampling to mitigate gender imbalance. |
| `Adult2_OVERsam.py`         | kNN on Adult dataset using SMOTE oversampling to balance gender. |
| `German1_UNDER.py`          | kNN on German Credit dataset using undersampling. |
| `Germ1_OVER.py`             | kNN on German Credit dataset using oversampling. |
| `novel.py`                  | Implements a **novel approach** with feature weighting + undersampling for fairness and accuracy. |
| `interactive_bar_plot.html`| Interactive visualization of gender-specific accuracy and overall model comparisons. |

#Results Summary
- Undersampling improved fairness with minor accuracy loss.
- SMOTE Oversampling** helped recover performance while reducing bias.
- Novel Method** showed:
  - Accuracy: 86.1%
  - Male Accuracy: 80.6%
  - Female Accuracy: 91.6%
  - Fairness gap reduction

#Conclusion
Bias-aware sampling and feature engineering significantly improve fairness in kNN classification. 
The novel approach balances both fairness and accuracy and is promising for real-world deployment.

#References
- Barocas et al. – *Fairness and Machine Learning*
- Dwork et al. – *Fairness Through Awareness*
- Kamiran & Calders – *Data Preprocessing Techniques*
- Hort et al. (2021) – *Bias Mitigation Survey*

