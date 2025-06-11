
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('adult/adult.data', header=None)

# Define column names
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
    'marital_status', 'occupation', 'relationship', 'race', 
    'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
data.columns = column_names

# Convert income to binary label
data['income'] = data['income'].str.strip()
data['income'] = (data['income'] == '>50K').astype(int)

# Handle missing values in 'sex' column
data['sex'] = data['sex'].replace('?', np.nan)
data.dropna(subset=['sex'], inplace=True)

# Handle leading/trailing spaces in the 'sex' column
data['sex'] = data['sex'].str.strip()

# Convert categorical columns to numerical form using get_dummies
data = pd.get_dummies(data, columns=[
    'workclass', 'education', 'marital_status', 
    'occupation', 'relationship', 'race', 'sex', 
    'native_country'
])

# Count the number of males and females in the dataset
gender_counts = data[['sex_Female', 'sex_Male']].sum()
print("Number of Males:", gender_counts['sex_Male'])
print("Number of Females:", gender_counts['sex_Female'])

# Create a balanced dataset by reducing the number of males
males = data[data['sex_Male'] == 1]
females = data[data['sex_Female'] == 1]

# Randomly sample males to match the number of females for balance
males_sampled = males.sample(n=len(females), random_state=42)

# Combine the sampled males with the females to form a balanced dataset
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the balanced dataset
X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

# Define the range of k values for KNN
k_values = range(1, 21)

# K-Fold Cross Validation for fairness-unaware model (trained on balanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in k_values:
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_balanced):
        X_train_fold, X_test_fold = X_balanced[train_index], X_balanced[test_index]
        y_train_fold, y_test_fold = y_balanced[train_index], y_balanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_unaware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\nBest k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Separate features and labels for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train Fairness-Unaware model on the balanced dataset
model_unaware = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware.fit(X_train, y_train)
y_pred_unaware = model_unaware.predict(X_test)
accuracy_unaware = accuracy_score(y_test, y_pred_unaware)

# Print Fairness-Unaware model accuracy
print("\nFairness-Unaware Model Accuracy:", accuracy_unaware)

# Calculate accuracies for Male and Female groups (disaggregated by gender)
accuracies_sex_unaware = []
sex_labels = ['Male', 'Female']

for sex in sex_labels:
    sex_column = 'sex_' + sex
    if sex_column in balanced_data.columns:
        sex_mask = X_test[:, balanced_data.columns.get_loc(sex_column)] == 1
        y_true = y_test[sex_mask]

        if y_true.size > 0:
            y_pred_unaware_sex = y_pred_unaware[sex_mask]
            accuracy_unaware_sex = accuracy_score(y_true, y_pred_unaware_sex)
            accuracies_sex_unaware.append(accuracy_unaware_sex)
        else:
            accuracies_sex_unaware.append(0)

# Print gender-specific accuracy for the Fairness-Unaware model
print("\nFairness-Unaware Model - Male Accuracy:", accuracies_sex_unaware[0])
print("Fairness-Unaware Model - Female Accuracy:", accuracies_sex_unaware[1])

# Fairness-Aware Model: KNN with fairness-focused adjustments
model_fairness_aware = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_fairness_aware.fit(X_train, y_train)
y_pred_fairness_aware = model_fairness_aware.predict(X_test)
accuracy_fairness_aware = accuracy_score(y_test, y_pred_fairness_aware)

# Print Fairness-Aware model accuracy
print("\nFairness-Aware Accuracy:", accuracy_fairness_aware)

# Calculate accuracies for Male and Female groups (disaggregated by gender) for the Fairness-Aware model
accuracies_sex_fairness_aware = []

for sex in sex_labels:
    sex_column = 'sex_' + sex
    if sex_column in balanced_data.columns:
        sex_mask = X_test[:, balanced_data.columns.get_loc(sex_column)] == 1
        y_true = y_test[sex_mask]

        if y_true.size > 0:
            y_pred_fairness_aware_sex = y_pred_fairness_aware[sex_mask]
            accuracy_fairness_aware_sex = accuracy_score(y_true, y_pred_fairness_aware_sex)
            accuracies_sex_fairness_aware.append(accuracy_fairness_aware_sex)
        else:
            accuracies_sex_fairness_aware.append(0)

# Print gender-specific accuracy for the Fairness-Aware model
print("\nFairness-Aware Model - Male Accuracy:", accuracies_sex_fairness_aware[0])
print("Fairness-Aware Model - Female Accuracy:", accuracies_sex_fairness_aware[1])

# Plotting the accuracies by gender for the fairness-aware model
if accuracies_sex_fairness_aware:
    x = np.arange(len(sex_labels))
    width = 0.3

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, accuracies_sex_unaware, width, label='Fairness-Unaware')
    bars2 = ax.bar(x + width/2, accuracies_sex_fairness_aware, width, label='Fairness-Aware')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Gender and Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels(sex_labels)
    ax.legend()

    plt.show()
else:
    print("No valid accuracies to plot.")



