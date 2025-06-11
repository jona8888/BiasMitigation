#Bias Mitigation in k-Nearest Neighbors (kNN)

This project explores how to reduce gender bias in k-Nearest Neighbors (kNN) classification models using fairness-focused preprocessing techniques. The aim is to improve fairness without sacrificing too much accuracy.

#Objective

We test several ways to make kNN predictions fairer, including:
- **Undersampling**: Reducing the size of the overrepresented group
- **Oversampling (SMOTE)**: Generating synthetic examples for the underrepresented group
- **Feature weighting**: Giving more importance to features like education and hours worked

#Datasets

- **Adult Census Income**: Predicts whether an individual's income is over $50K based on demographics and work history
- **German Credit**: Assesses creditworthiness using financial and personal information

#Methodology

1. **Baseline**: Standard kNN model without any fairness tweaks (`knn682.py`)
2. **Sampling methods**:
   - **Undersampling**: Drops some male samples to balance the gender ratio
   - **Oversampling**: Uses SMOTE to generate synthetic female samples
3. **Novel technique**:
   - Combines feature weighting and undersampling for better fairness and accuracy

#Project Structure

| File                        | Description |
|-----------------------------|-------------|
| `knn682.py`                 | Baseline model on the Adult dataset (no mitigation) |
| `Adult1_UNDERsam.py`        | Adult dataset with undersampling applied |
| `Adult2_OVERsam.py`         | Adult dataset with SMOTE oversampling |
| `German1_UNDER.py`          | German Credit dataset with undersampling |
| `Germ1_OVER.py`             | German Credit dataset with oversampling |
| `novel.py`                  | Novel method with custom feature weighting |
| `interactive_bar_plot.html`| Visualization comparing gender-based accuracy across models |

#Results Summary

- **Undersampling** helped close the gender accuracy gap, though it slightly reduced overall accuracy.
- **SMOTE oversampling** maintained higher accuracy while reducing bias.
- **The novel method** produced the best balance:
  - Overall Accuracy: 86.1%
  - Male Accuracy: 80.6%
  - Female Accuracy: 91.6%

#Conclusion

Preprocessing strategies like sampling and feature weighting can improve fairness in kNN models. Our custom method shows that it's possible to make fairer predictions without heavily compromising accuracy.

#References

- Barocas et al. – *Fairness and Machine Learning*
- Dwork et al. – *Fairness Through Awareness*
- Kamiran & Calders – *Data Preprocessing Techniques*
- Hort et al. (2021) – *Bias Mitigation Survey*
