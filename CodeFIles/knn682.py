#STRAT K FOLD
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("adult/adult.data")

# Preprocess the dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country', 'income'
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

# Check the number of males and females in the original dataset
original_males = data[data['sex_Male'] == 1]
original_females = data[data['sex_Female'] == 1]
print(f"Original dataset - Number of males: {original_males.shape[0]}, Number of females: {original_females.shape[0]}")

# Create a balanced dataset by reducing the number of males
males = data[data['sex_Male'] == 1]
females = data[data['sex_Female'] == 1]

# Randomly sample 10,000 males
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the number of males and females in the balanced dataset
balanced_males = balanced_data[balanced_data['sex_Male'] == 1]
balanced_females = balanced_data[balanced_data['sex_Female'] == 1]
print(f"Balanced dataset - Number of males: {balanced_males.shape[0]}, Number of females: {balanced_females.shape[0]}")

# Separate features and labels for the balanced data
X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

# Stratified K-Fold Cross Validation for Fairness-Aware Model
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_k_fairness_aware = None
best_accuracy_fairness_aware = 0
k_values = range(1, 21)

for k in k_values:
    model_fairness_aware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []

    for train_index, test_index in kf.split(X_balanced, y_balanced):
        X_train_fold, X_test_fold = X_balanced[train_index], X_balanced[test_index]
        y_train_fold, y_test_fold = y_balanced[train_index], y_balanced[test_index]

        model_fairness_aware_k.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fairness_aware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))

    avg_accuracy = np.mean(fold_accuracies)

    if avg_accuracy > best_accuracy_fairness_aware:
        best_accuracy_fairness_aware = avg_accuracy
        best_k_fairness_aware = k

print(f'Best k for Fairness-Aware Model with Stratified K-Fold: {best_k_fairness_aware}')
print(f'Best Accuracy for Fairness-Aware Model with Stratified K-Fold: {best_accuracy_fairness_aware}')

# Final Model Evaluation for Fairness-Aware Model with the best k
model_fairness_aware = KNeighborsClassifier(n_neighbors=best_k_fairness_aware)
model_fairness_aware.fit(X_balanced, y_balanced)
y_pred_fairness_aware = model_fairness_aware.predict(X_balanced)
accuracy_fairness_aware = accuracy_score(y_balanced, y_pred_fairness_aware)

# Print accuracy of the Fairness-Aware model
print(f'Fairness-Aware Model Accuracy with Stratified K-Fold: {accuracy_fairness_aware}')

# For Fairness-Unaware model, we perform similar steps but without balancing
# Recreate the feature set and labels for the original data (unbalanced)
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Stratified K-Fold Cross Validation for Fairness-Unaware Model
best_k_fairness_unaware = None
best_accuracy_fairness_unaware = 0

for k in k_values:
    model_fairness_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []

    for train_index, test_index in kf.split(X_unbalanced, y_unbalanced):
        X_train_fold, X_test_fold = X_unbalanced[train_index], X_unbalanced[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]

        model_fairness_unaware_k.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fairness_unaware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))

    avg_accuracy = np.mean(fold_accuracies)

    if avg_accuracy > best_accuracy_fairness_unaware:
        best_accuracy_fairness_unaware = avg_accuracy
        best_k_fairness_unaware = k

print(f'Best k for Fairness-Unaware Model with Stratified K-Fold: {best_k_fairness_unaware}')
print(f'Best Accuracy for Fairness-Unaware Model with Stratified K-Fold: {best_accuracy_fairness_unaware}')

# Final Model Evaluation for Fairness-Unaware Model with the best k
model_fairness_unaware = KNeighborsClassifier(n_neighbors=best_k_fairness_unaware)
model_fairness_unaware.fit(X_unbalanced, y_unbalanced)
y_pred_fairness_unaware = model_fairness_unaware.predict(X_unbalanced)
accuracy_fairness_unaware = accuracy_score(y_unbalanced, y_pred_fairness_unaware)

# Print accuracy of the Fairness-Unaware model
print(f'Fairness-Unaware Model Accuracy with Stratified K-Fold: {accuracy_fairness_unaware}')

# Male and female accuracies for Fairness-Aware Model
male_accuracy_fairness_aware = accuracy_score(y_balanced[balanced_data['sex_Male'] == 1], y_pred_fairness_aware[balanced_data['sex_Male'] == 1])
female_accuracy_fairness_aware = accuracy_score(y_balanced[balanced_data['sex_Female'] == 1], y_pred_fairness_aware[balanced_data['sex_Female'] == 1])

print(f'Fairness-Aware Model Accuracy for Males: {male_accuracy_fairness_aware}')
print(f'Fairness-Aware Model Accuracy for Females: {female_accuracy_fairness_aware}')

# Male and female accuracies for Fairness-Unaware Model
male_accuracy_fairness_unaware = accuracy_score(y_unbalanced[data['sex_Male'] == 1], y_pred_fairness_unaware[data['sex_Male'] == 1])
female_accuracy_fairness_unaware = accuracy_score(y_unbalanced[data['sex_Female'] == 1], y_pred_fairness_unaware[data['sex_Female'] == 1])

print(f'Fairness-Unaware Model Accuracy for Males: {male_accuracy_fairness_unaware}')
print(f'Fairness-Unaware Model Accuracy for Females: {female_accuracy_fairness_unaware}')
'''
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE  # If you want to apply SMOTE

# Load dataset
data = pd.read_csv("adult/adult.data")

# Preprocess the dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country', 'income'
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

# Create a balanced dataset by reducing the number of males
males = data[data['sex_Male'] == 1]
females = data[data['sex_Female'] == 1]

# Randomly sample 10,000 males
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels for the balanced data
X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize SMOTE (optional, if you want to apply SMOTE for oversampling)
smote = SMOTE(random_state=42)

# Variables to track the best model
best_k = None
best_accuracy = 0
k_values = range(1, 21)

# Regular K-Fold Cross Validation for Fairness-Unaware Model (with SMOTE)
for k in k_values:
    model_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []

    # Loop through each fold
    for train_index, test_index in kf.split(X_balanced):
        X_train_fold, X_test_fold = X_balanced[train_index], X_balanced[test_index]
        y_train_fold, y_test_fold = y_balanced[train_index], y_balanced[test_index]

        # Apply SMOTE to the training data (only on training set)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)

        # Train the model on the resampled data
        model_k.fit(X_train_smote, y_train_smote)
        
        # Make predictions on the test set
        y_pred_fold = model_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))

    # Calculate the average accuracy for this value of k
    avg_accuracy = np.mean(fold_accuracies)

    # Track the best k value based on the highest accuracy
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k

print(f'Best k for Regular K-Fold Model with SMOTE: {best_k}')
print(f'Best Accuracy for Regular K-Fold Model with SMOTE: {best_accuracy}')

# Final Model Evaluation for Regular K-Fold Model with the best k
model_best_k = KNeighborsClassifier(n_neighbors=best_k)
model_best_k.fit(X_balanced, y_balanced)
y_pred_best_k = model_best_k.predict(X_balanced)
accuracy_best_k = accuracy_score(y_balanced, y_pred_best_k)

# Print accuracy of the best model
print(f'Final Model Accuracy with Regular K-Fold: {accuracy_best_k}')
'''
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE  # Optional, if you want to apply SMOTE

# Load dataset
data = pd.read_csv("adult/adult.data")

# Preprocess the dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country', 'income'
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

# Check the number of males and females in the original dataset
original_males = data[data['sex_Male'] == 1]
original_females = data[data['sex_Female'] == 1]
print(f"Original dataset - Number of males: {original_males.shape[0]}, Number of females: {original_females.shape[0]}")

# Create a balanced dataset by reducing the number of males
males = data[data['sex_Male'] == 1]
females = data[data['sex_Female'] == 1]

# Randomly sample 10,000 males
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the number of males and females in the balanced dataset
balanced_males = balanced_data[balanced_data['sex_Male'] == 1]
balanced_females = balanced_data[balanced_data['sex_Female'] == 1]
print(f"Balanced dataset - Number of males: {balanced_males.shape[0]}, Number of females: {balanced_females.shape[0]}")

# Separate features and labels for the balanced data
X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

# Separate features and labels for the unbalanced data (original data)
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Initialize K-Fold cross-validation (no StratifiedKFold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize SMOTE (optional, if you want to apply SMOTE for oversampling)
smote = SMOTE(random_state=42)

# Variables to track the best model for the Fairness-Unaware Model
best_k_unaware = None
best_accuracy_unaware = 0
k_values = range(1, 21)

# Regular K-Fold Cross Validation for the Fairness-Unaware Model (using unbalanced data)
for k in k_values:
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []

    for train_index, test_index in kf.split(X_unbalanced):
        X_train_fold, X_test_fold = X_unbalanced[train_index], X_unbalanced[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]

        # Train the model on the unbalanced data
        model_unaware_k.fit(X_train_fold, y_train_fold)
        
        # Make predictions on the test set
        y_pred_fold = model_unaware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))

    avg_accuracy = np.mean(fold_accuracies)

    # Track the best k value based on the highest accuracy
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print(f'Best k for Fairness-Unaware Model with Regular K-Fold: {best_k_unaware}')
print(f'Best Accuracy for Fairness-Unaware Model with Regular K-Fold: {best_accuracy_unaware}')

# Final Model Evaluation for the Fairness-Unaware Model with the best k
model_unaware_best_k = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best_k.fit(X_unbalanced, y_unbalanced)
y_pred_unaware_best_k = model_unaware_best_k.predict(X_unbalanced)
accuracy_unaware_best_k = accuracy_score(y_unbalanced, y_pred_unaware_best_k)

# Print accuracy of the Fairness-Unaware model
print(f'Fairness-Unaware Model Accuracy with Regular K-Fold: {accuracy_unaware_best_k}')

# Male and female accuracies for the Fairness-Unaware model
male_accuracy_unaware = accuracy_score(y_unbalanced[data['sex_Male'] == 1], 
                                       y_pred_unaware_best_k[data['sex_Male'] == 1])
female_accuracy_unaware = accuracy_score(y_unbalanced[data['sex_Female'] == 1], 
                                         y_pred_unaware_best_k[data['sex_Female'] == 1])

print(f'Fairness-Unaware Model Accuracy for Males: {male_accuracy_unaware}')
print(f'Fairness-Unaware Model Accuracy for Females: {female_accuracy_unaware}')

# Variables to track the best k for the Fairness-Aware Model
best_k_aware = None
best_accuracy_aware = 0

# Regular K-Fold Cross Validation for the Fairness-Aware Model (using balanced data)
for k in k_values:
    model_aware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []

    for train_index, test_index in kf.split(X_balanced):
        X_train_fold, X_test_fold = X_balanced[train_index], X_balanced[test_index]
        y_train_fold, y_test_fold = y_balanced[train_index], y_balanced[test_index]

        # Apply SMOTE to the training data (only on training set)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)

        # Train the model on the resampled data
        model_aware_k.fit(X_train_smote, y_train_smote)
        
        # Make predictions on the test set
        y_pred_fold = model_aware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))

    avg_accuracy = np.mean(fold_accuracies)

    # Track the best k value based on the highest accuracy
    if avg_accuracy > best_accuracy_aware:
        best_accuracy_aware = avg_accuracy
        best_k_aware = k

print(f'Best k for Fairness-Aware Model with Regular K-Fold: {best_k_aware}')
print(f'Best Accuracy for Fairness-Aware Model with Regular K-Fold: {best_accuracy_aware}')

# Final Model Evaluation for the Fairness-Aware Model with the best k
model_aware_best_k = KNeighborsClassifier(n_neighbors=best_k_aware)
model_aware_best_k.fit(X_balanced, y_balanced)
y_pred_aware_best_k = model_aware_best_k.predict(X_balanced)
accuracy_aware_best_k = accuracy_score(y_balanced, y_pred_aware_best_k)

# Print accuracy of the Fairness-Aware model
print(f'Fairness-Aware Model Accuracy with Regular K-Fold: {accuracy_aware_best_k}')

# Male and female accuracies for the Fairness-Aware model
male_accuracy_aware = accuracy_score(y_balanced[balanced_data['sex_Male'] == 1], 
                                     y_pred_aware_best_k[balanced_data['sex_Male'] == 1])
female_accuracy_aware = accuracy_score(y_balanced[balanced_data['sex_Female'] == 1], 
                                       y_pred_aware_best_k[balanced_data['sex_Female'] == 1])

print(f'Fairness-Aware Model Accuracy for Males: {male_accuracy_aware}')
print(f'Fairness-Aware Model Accuracy for Females: {female_accuracy_aware}')
'''
'''
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

# Randomly sample 10,000 males
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original (unprocessed) data
X_original = data.drop(columns=['income']).values
y_original = data['income'].values

# K-Fold Cross Validation for Fairness-Unaware model (using original, unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_values = range(1, 21)  # Test k values from 1 to 20
best_k_unaware = None
best_accuracy_unaware = 0

for k in k_values:
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_original):
        X_train_fold, X_test_fold = X_original[train_index], X_original[test_index]
        y_train_fold, y_test_fold = y_original[train_index], y_original[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_unaware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Aware Baseline: Preprocessing the data for sex
sex_columns = [col for col in balanced_data.columns if 'sex_' in col]
if not sex_columns:
    raise KeyError("No sex columns found after dummy encoding.")

grouped_data_sex = balanced_data.groupby(sex_columns)

# Sample the data to ensure equal representation for each sex category
balanced_data_sex = pd.DataFrame()

for _, group in grouped_data_sex:
    balanced_data_sex = pd.concat([balanced_data_sex, group.sample(n=min(len(group), 2000), random_state=42)])

balanced_data_sex.reset_index(drop=True, inplace=True)

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data_sex.drop(columns=['income']).values
y_balanced_sex = balanced_data_sex['income'].values

# K-Fold Cross Validation for Fairness-Aware model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_fairness_aware = None
best_accuracy_fairness_aware = 0

for k in k_values:
    model_fairness_aware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_balanced_sex):
        X_train_fold, X_test_fold = X_balanced_sex[train_index], X_balanced_sex[test_index]
        y_train_fold, y_test_fold = y_balanced_sex[train_index], y_balanced_sex[test_index]
        
        model_fairness_aware_k.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fairness_aware_k.predict(X_test_fold)
        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_fairness_aware:
        best_accuracy_fairness_aware = avg_accuracy
        best_k_fairness_aware = k

print("Best k for Fairness-Aware (Sex):", best_k_fairness_aware)
print("Best Accuracy for Fairness-Aware (Sex):", best_accuracy_fairness_aware)

# Split the balanced dataset into training and testing sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced_sex, y_balanced_sex, test_size=0.2, random_state=42)

# Fairness-Unaware model using best k (trained on original data)
model_unaware = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware.fit(X_original, y_original)  # Train on the original, unbalanced data
y_pred_unaware = model_unaware.predict(X_original)  # Using the original data for predictions
accuracy_unaware = accuracy_score(y_original, y_pred_unaware)

print("Fairness-Unaware Baseline Accuracy:", accuracy_unaware)

# Fairness-Aware model using best k (trained on balanced data)
model_fairness_aware = KNeighborsClassifier(n_neighbors=best_k_fairness_aware)
model_fairness_aware.fit(X_train_balanced, y_train_balanced)
y_pred_fairness_aware = model_fairness_aware.predict(X_test_balanced)
accuracy_fairness_aware = accuracy_score(y_test_balanced, y_pred_fairness_aware)

print("Fairness-Aware Baseline Accuracy (Sex):", accuracy_fairness_aware)

# Calculate accuracies for Male and Female groups
sex_labels = ['Male', 'Female']
accuracies_sex = []

# Create masks for test set predictions
for sex in sex_labels:
    sex_column = 'sex_' + sex
    if sex_column in balanced_data_sex.columns:
        sex_mask = X_test_balanced[:, balanced_data_sex.columns.get_loc(sex_column)] == 1
        y_true = y_test_balanced[sex_mask]

        if y_true.size > 0:
            y_pred_unaware_sex = y_pred_unaware[np.isin(X_original[:, balanced_data_sex.columns.get_loc(sex_column)], [1])][:len(y_true)]
            y_pred_fairness_aware_sex = y_pred_fairness_aware[sex_mask]

            accuracy_unaware_sex = accuracy_score(y_true, y_pred_unaware_sex)
            accuracy_fairness_aware_sex = accuracy_score(y_true, y_pred_fairness_aware_sex)

            accuracies_sex.append((accuracy_unaware_sex, accuracy_fairness_aware_sex))
            print(f"{sex} Fairness-Unaware Accuracy:", accuracy_unaware_sex)
            print(f"{sex} Fairness-Aware Accuracy:", accuracy_fairness_aware_sex)
        else:
            accuracies_sex.append((0, 0))
    else:
        accuracies_sex.append((0, 0))

# Plotting
if any(acc[0] != 0 or acc[1] != 0 for acc in accuracies_sex):
    x = np.arange(len(sex_labels))
    width = 0.31

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, [acc[0] for acc in accuracies_sex], width, label='Fairness-Unaware')
    bars2 = ax.bar(x + width/2, [acc[1] for acc in accuracies_sex], width, label='Fairness-Aware (Sex)')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Gender and Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels(sex_labels)
    ax.legend()

    plt.show()
else:
    print("No valid accuracies to plot.")
'''
'''
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

# Randomly sample 10,000 males
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data.drop(columns=['income']).values
y_balanced_sex = balanced_data['income'].values

# Fairness-Unaware model training without LFR
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

# Modify training loop to incorporate fairness regularization for the unaware model
for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced):  # Use the unbalanced data
        X_train_fold, X_test_fold = X_unbalanced[train_index], X_unbalanced[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        # Train the model
        model_unaware_k.fit(X_train_fold, y_train_fold)
        
        # Evaluate accuracy on the fold
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Calculate fairness-unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# Now training the fairness-aware model on the balanced data
model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced_sex, y_balanced_sex)

# Calculate accuracy and fairness on balanced dataset
y_pred_aware = model_aware_best.predict(X_balanced_sex)
overall_accuracy_aware = accuracy_score(y_balanced_sex, y_pred_aware)

male_mask_aware = balanced_data['sex_Male'] == 1
female_mask_aware = balanced_data['sex_Female'] == 1

male_accuracy_aware = accuracy_score(y_balanced_sex[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced_sex[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)
'''

'''
#backup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Randomly sample 10,000 males to match the number of females
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data.drop(columns=['income']).values
y_balanced_sex = balanced_data['income'].values

# Scale the features using StandardScaler
scaler = StandardScaler()
X_unbalanced_scaled = scaler.fit_transform(X_unbalanced)
X_balanced_sex_scaled = scaler.transform(X_balanced_sex)

# Initialize feature weights (e.g., give higher weights to certain features)
# Let's assume we want to apply weights to the first 26 features
feature_weights = np.ones(X_balanced_sex_scaled.shape[1])  # 108 features by default, all weights set to 1
# If you want to apply weights to only specific features (e.g., the first 26), update this
feature_weights[:26] = 2  # Example: Apply a weight of 2 to the first 26 features

# Apply the feature weights to the scaled dataset
X_balanced_sex_weighted = X_balanced_sex_scaled * feature_weights

# K-Fold Cross Validation for Fairness-Unaware model (using unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced_scaled):  # Use the unbalanced data
        X_train_fold, X_test_fold = X_unbalanced_scaled[train_index], X_unbalanced_scaled[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced_scaled, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced_scaled)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# Now training the fairness-aware model on the weighted data (balanced dataset)
model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced_sex_weighted, y_balanced_sex)

# Fairness-Aware accuracy for different groups
y_pred_aware = model_aware_best.predict(X_balanced_sex_weighted)
overall_accuracy_aware = accuracy_score(y_balanced_sex, y_pred_aware)

male_mask_aware = balanced_data['sex_Male'] == 1
female_mask_aware = balanced_data['sex_Female'] == 1

male_accuracy_aware = accuracy_score(y_balanced_sex[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced_sex[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)
'''

'''
#new backup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Randomly sample 10,000 males to match the number of females
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data.drop(columns=['income']).values
y_balanced_sex = balanced_data['income'].values

# Scale the features using StandardScaler
scaler = StandardScaler()
X_unbalanced_scaled = scaler.fit_transform(X_unbalanced)
X_balanced_sex_scaled = scaler.transform(X_balanced_sex)

# Initialize feature weights (e.g., give higher weights to certain features)
feature_weights = np.ones(X_balanced_sex_scaled.shape[1])  # 108 features by default, all weights set to 1
# If you want to apply weights to only specific features (e.g., the first 26), update this
feature_weights[:26] = 1.5  # Example: Apply a weight of 1.5 to the first 26 features

# Apply the feature weights to the scaled dataset
X_balanced_sex_weighted = X_balanced_sex_scaled * feature_weights

# K-Fold Cross Validation for Fairness-Unaware model (using unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced_scaled):  # Use the unbalanced data
        X_train_fold, X_test_fold = X_unbalanced_scaled[train_index], X_unbalanced_scaled[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced_scaled, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced_scaled)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# Now training the fairness-aware model on the weighted data (balanced dataset)
model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced_sex_weighted, y_balanced_sex)

# Use K-Fold Cross-Validation for Fairness-Aware model (on the weighted balanced data)
cv_scores_aware = cross_val_score(model_aware_best, X_balanced_sex_weighted, y_balanced_sex, cv=kf)
print("Fairness-Aware Cross-validation scores:", cv_scores_aware)
print("Fairness-Aware Average cross-validation score:", np.mean(cv_scores_aware))

# Fairness-Aware accuracy for different groups
y_pred_aware = model_aware_best.predict(X_balanced_sex_weighted)
overall_accuracy_aware = accuracy_score(y_balanced_sex, y_pred_aware)

male_mask_aware = balanced_data['sex_Male'] == 1
female_mask_aware = balanced_data['sex_Female'] == 1

male_accuracy_aware = accuracy_score(y_balanced_sex[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced_sex[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)


'''

#new

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Randomly sample 10,000 males to match the number of females
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data.drop(columns=['income']).values
y_balanced_sex = balanced_data['income'].values

# Scale the features using StandardScaler
scaler = StandardScaler()
X_unbalanced_scaled = scaler.fit_transform(X_unbalanced)
X_balanced_sex_scaled = scaler.transform(X_balanced_sex)

# Initialize feature weights (e.g., give higher weights to certain features)
feature_weights = np.ones(X_balanced_sex_scaled.shape[1])  # 108 features by default, all weights set to 1

# Apply a weight of 10 to education-related features
education_columns = [i for i in range(26, 45)]  # Adjust based on actual column indices for education features
feature_weights[education_columns] = 10.0  # Apply weight of 10 to all education-related features

# Apply a weight of 10 to occupation-related features
occupation_columns = [i for i in range(45, 64)]  # Adjust based on actual column indices for occupation features
feature_weights[occupation_columns] = 10.0  # Apply weight of 10 to all occupation-related features

# Apply a weight of 10 to hours worked per week
feature_weights[63] = 10.0  # Apply weight of 10 to hours_per_week (assuming it's at column index 63)

# Apply a weight of 10 to capital gain and capital loss features
feature_weights[10] = 10.0  # Apply weight of 10 to capital gain (assuming it's at column index 10)
feature_weights[11] = 10.0  # Apply weight of 10 to capital loss (assuming it's at column index 11)

# Apply a weight of 10 to the age feature
feature_weights[0] = 10.0  # Apply weight of 10 to age (assuming it's at column index 0)

# Apply these weights to the scaled dataset for Fairness-Aware model
X_balanced_sex_weighted = X_balanced_sex_scaled * feature_weights

# --- Fairness-Unaware Model --- 
# (using unbalanced raw data)

# Initialize K-Fold Cross Validation for Fairness-Unaware model (using unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced):  # Use the unbalanced data
        X_train_fold, X_test_fold = X_unbalanced[train_index], X_unbalanced[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# --- Fairness-Aware Model --- 
# (using balanced weighted data)

model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced_sex_weighted, y_balanced_sex)

# Use K-Fold Cross-Validation for Fairness-Aware model (on the weighted balanced data)
cv_scores_aware = cross_val_score(model_aware_best, X_balanced_sex_weighted, y_balanced_sex, cv=kf)
print("Fairness-Aware Cross-validation scores:", cv_scores_aware)
print("Fairness-Aware Average cross-validation score:", np.mean(cv_scores_aware))

# Fairness-Aware accuracy for different groups
y_pred_aware = model_aware_best.predict(X_balanced_sex_weighted)
overall_accuracy_aware = accuracy_score(y_balanced_sex, y_pred_aware)

male_mask_aware = balanced_data['sex_Male'] == 1
female_mask_aware = balanced_data['sex_Female'] == 1

male_accuracy_aware = accuracy_score(y_balanced_sex[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced_sex[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)
'''
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Randomly sample 10,000 males to match the number of females
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data.drop(columns=['income']).values
y_balanced_sex = balanced_data['income'].values

# Scale the features using StandardScaler
scaler = StandardScaler()
X_unbalanced_scaled = scaler.fit_transform(X_unbalanced)
X_balanced_sex_scaled = scaler.transform(X_balanced_sex)

# Initialize feature weights (e.g., give higher weights to certain features)
feature_weights = np.ones(X_balanced_sex_scaled.shape[1])  # 108 features by default, all weights set to 1

# Apply a weight of 10 to education-related features
education_columns = [i for i in range(26, 45)]  # Adjust based on actual column indices for education features
feature_weights[education_columns] = 10.0  # Apply weight of 10 to all education-related features

# Apply a weight of 10 to occupation-related features
occupation_columns = [i for i in range(45, 64)]  # Adjust based on actual column indices for occupation features
feature_weights[occupation_columns] = 10.0  # Apply weight of 10 to all occupation-related features

# Apply a weight of 10 to hours worked per week
feature_weights[63] = 10.0  # Apply weight of 10 to hours_per_week (assuming it's at column index 63)

# Apply a weight of 10 to capital gain and capital loss features
feature_weights[10] = 10.0  # Apply weight of 10 to capital gain (assuming it's at column index 10)
feature_weights[11] = 10.0  # Apply weight of 10 to capital loss (assuming it's at column index 11)

# Apply a weight of 10 to the age feature
feature_weights[0] = 10.0  # Apply weight of 10 to age (assuming it's at column index 0)

# Apply these weights to the scaled dataset for Fairness-Aware model
X_balanced_sex_weighted = X_balanced_sex_scaled * feature_weights

# --- Fairness-Unaware Model --- 
# (using unbalanced raw data)

# Initialize K-Fold Cross Validation for Fairness-Unaware model (using unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced):  # Use the unbalanced data
        X_train_fold, X_test_fold = X_unbalanced[train_index], X_unbalanced[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# --- Fairness-Aware Model --- 
# (using balanced weighted data)

model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced_sex_weighted, y_balanced_sex)

# Use K-Fold Cross-Validation for Fairness-Aware model (on the weighted balanced data)
cv_scores_aware = cross_val_score(model_aware_best, X_balanced_sex_weighted, y_balanced_sex, cv=kf)
print("Fairness-Aware Cross-validation scores:", cv_scores_aware)
print("Fairness-Aware Average cross-validation score:", np.mean(cv_scores_aware))

# Fairness-Aware accuracy for different groups
y_pred_aware = model_aware_best.predict(X_balanced_sex_weighted)
overall_accuracy_aware = accuracy_score(y_balanced_sex, y_pred_aware)

male_mask_aware = balanced_data['sex_Male'] == 1
female_mask_aware = balanced_data['sex_Female'] == 1

male_accuracy_aware = accuracy_score(y_balanced_sex[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced_sex[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)'''


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Randomly sample 10,000 males to match the number of females
males_sampled = males.sample(n=10771, random_state=42)

# Combine the sampled males with the females
balanced_data = pd.concat([males_sampled, females])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Count the new number of males and females after balancing
new_gender_counts = balanced_data[['sex_Female', 'sex_Male']].sum()
print("New Number of Males:", new_gender_counts['sex_Male'])
print("New Number of Females:", new_gender_counts['sex_Female'])

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Separate features and labels for the balanced data
X_balanced_sex = balanced_data.drop(columns=['income']).values
y_balanced_sex = balanced_data['income'].values

# Scale the features using StandardScaler
scaler = StandardScaler()
X_unbalanced_scaled = scaler.fit_transform(X_unbalanced)
X_balanced_sex_scaled = scaler.transform(X_balanced_sex)

# Initialize feature weights (e.g., give higher weights to certain features)
feature_weights = np.ones(X_balanced_sex_scaled.shape[1])  # 108 features by default, all weights set to 1
# If you want to apply weights to only specific features (e.g., the first 26), update this
feature_weights[:26] = 1.5  # Example: Apply a weight of 1.5 to the first 26 features

# Apply the feature weights to the scaled dataset
X_balanced_sex_weighted = X_balanced_sex_scaled * feature_weights

# K-Fold Cross Validation for Fairness-Unaware model (using unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced_scaled):  # Use the unbalanced data
        X_train_fold, X_test_fold = X_unbalanced_scaled[train_index], X_unbalanced_scaled[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced_scaled, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced_scaled)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# Now training the fairness-aware model on the weighted data (balanced dataset)
model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced_sex_weighted, y_balanced_sex)

# Use K-Fold Cross-Validation for Fairness-Aware model (on the weighted balanced data)
cv_scores_aware = cross_val_score(model_aware_best, X_balanced_sex_weighted, y_balanced_sex, cv=kf)
print("Fairness-Aware Cross-validation scores:", cv_scores_aware)
print("Fairness-Aware Average cross-validation score:", np.mean(cv_scores_aware))

# Fairness-Aware accuracy for different groups
y_pred_aware = model_aware_best.predict(X_balanced_sex_weighted)
overall_accuracy_aware = accuracy_score(y_balanced_sex, y_pred_aware)

male_mask_aware = balanced_data['sex_Male'] == 1
female_mask_aware = balanced_data['sex_Female'] == 1

male_accuracy_aware = accuracy_score(y_balanced_sex[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced_sex[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)
'''

#FINAL SMOTE
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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

# Separate features and labels for the original unbalanced data
X_unbalanced = data.drop(columns=['income']).values
y_unbalanced = data['income'].values

# Scale the features using StandardScaler
scaler = StandardScaler()
X_unbalanced_scaled = scaler.fit_transform(X_unbalanced)

# Initialize feature weights (e.g., give higher weights to certain features)
feature_weights = np.ones(X_unbalanced_scaled.shape[1])  # 108 features by default, all weights set to 1
# If you want to apply weights to only specific features (e.g., the first 26), update this
feature_weights[:26] = 1.5  # Example: Apply a weight of 1.5 to the first 26 features

# Apply the feature weights to the scaled dataset
X_unbalanced_scaled_weighted = X_unbalanced_scaled * feature_weights

# SMOTE for balancing the dataset (instead of downsampling)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_unbalanced_scaled_weighted, y_unbalanced)

# Check the new class distribution
print(f"Original class distribution: {np.bincount(y_unbalanced)}")
print(f"Balanced class distribution: {np.bincount(y_balanced)}")

# K-Fold Cross Validation for Fairness-Unaware model (using unbalanced data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k_unaware = None
best_accuracy_unaware = 0

for k in range(1, 21):  # Test k values from 1 to 20
    model_unaware_k = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    
    for train_index, test_index in kf.split(X_unbalanced_scaled, y_unbalanced):  # Use KFold (not Stratified)
        X_train_fold, X_test_fold = X_unbalanced_scaled[train_index], X_unbalanced_scaled[test_index]
        y_train_fold, y_test_fold = y_unbalanced[train_index], y_unbalanced[test_index]
        
        model_unaware_k.fit(X_train_fold, y_train_fold)
        fold_accuracy = accuracy_score(y_test_fold, model_unaware_k.predict(X_test_fold))
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = np.mean(fold_accuracies)
    
    if avg_accuracy > best_accuracy_unaware:
        best_accuracy_unaware = avg_accuracy
        best_k_unaware = k

print("\n--- Fairness-Unaware Results ---")
print("Best k for Fairness-Unaware:", best_k_unaware)
print("Best Accuracy for Fairness-Unaware:", best_accuracy_unaware)

# Fairness-Unaware accuracy for different groups
model_unaware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware_best.fit(X_unbalanced_scaled, y_unbalanced)

# Overall Fairness-Unaware Accuracy
y_pred_unaware = model_unaware_best.predict(X_unbalanced_scaled)
overall_accuracy_unaware = accuracy_score(y_unbalanced, y_pred_unaware)

# Male and Female Fairness-Unaware Accuracy
male_mask_unaware = data['sex_Male'] == 1
female_mask_unaware = data['sex_Female'] == 1

male_accuracy_unaware = accuracy_score(y_unbalanced[male_mask_unaware], y_pred_unaware[male_mask_unaware])
female_accuracy_unaware = accuracy_score(y_unbalanced[female_mask_unaware], y_pred_unaware[female_mask_unaware])

print("Overall Fairness-Unaware Accuracy:", overall_accuracy_unaware)
print("Male Fairness-Unaware Accuracy:", male_accuracy_unaware)
print("Female Fairness-Unaware Accuracy:", female_accuracy_unaware)

# Now training the fairness-aware model on the balanced dataset (with SMOTE)
model_aware_best = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_aware_best.fit(X_balanced, y_balanced)

# Use K-Fold Cross-Validation for Fairness-Aware model (on the weighted balanced data)
cv_scores_aware = cross_val_score(model_aware_best, X_balanced, y_balanced, cv=kf)
print("Fairness-Aware Cross-validation scores:", cv_scores_aware)
print("Fairness-Aware Average cross-validation score:", np.mean(cv_scores_aware))

# Fairness-Aware accuracy for different groups
y_pred_aware = model_aware_best.predict(X_balanced)
overall_accuracy_aware = accuracy_score(y_balanced, y_pred_aware)

# Identify male and female indices in the balanced data
male_mask_aware = y_balanced == 1  # Assuming that class 1 corresponds to males
female_mask_aware = y_balanced == 0  # Assuming that class 0 corresponds to females

male_accuracy_aware = accuracy_score(y_balanced[male_mask_aware], y_pred_aware[male_mask_aware])
female_accuracy_aware = accuracy_score(y_balanced[female_mask_aware], y_pred_aware[female_mask_aware])

print("\n--- Fairness-Aware Results ---")
print("Overall Fairness-Aware Accuracy:", overall_accuracy_aware)
print("Male Fairness-Aware Accuracy:", male_accuracy_aware)
print("Female Fairness-Aware Accuracy:", female_accuracy_aware)
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imPipeline

# Column names for the Adult dataset (as it doesn't contain headers in the raw data)
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country', 'income'
]

# Load the dataset from the 'adult.data' file
data = pd.read_csv('adult/adult.data', names=columns, na_values=' ?', sep=',\s', engine='python')

# Display the first few rows of the dataset
print(data.head())

# Step 1: Preprocess the data

# Handle missing values
# Replace '?' with NaN and use SimpleImputer to fill missing values
# 'age', 'education_num', 'capital_gain', 'capital_loss', and 'hours_per_week' are numeric
# 'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', and 'native_country' are categorical

# Separate features (X) and target (y)
X = data.drop('income', axis=1)
y = data['income'].map({'<=50K': 0, '>50K': 1})  # Convert income to binary (0 for <=50K, 1 for >50K)

# Step 2: Create a preprocessing pipeline
numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

# Preprocessing for numeric data: Imputation (fill missing values with the median)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Preprocessing for categorical data: Imputation and OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Updated parameter
])

# Combine both transformers into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 3: Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Create the SMOTE pipeline
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Combine everything into a single pipeline: SMOTE + Preprocessing + Random Forest
pipeline = imPipeline([
    ('smote', smote),
    ('preprocessor', preprocessor),
    ('model', model)
])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train the model
pipeline.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = pipeline.predict(X_test)

# Step 8: Evaluate the model
print(classification_report(y_test, y_pred))
