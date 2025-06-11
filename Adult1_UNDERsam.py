
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

# Separate features and labels for the balanced data
X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

# K-Fold Cross Validation for Fairness-Unaware model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_values = range(1, 21)  # Test k values from 1 to 20
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

# Fairness-Unaware model using best k
model_unaware = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware.fit(X_balanced, y_balanced)
y_pred_unaware = model_unaware.predict(X_balanced)  # Using the balanced data for predictions
accuracy_unaware = accuracy_score(y_balanced, y_pred_unaware)

print("Fairness-Unaware Baseline Accuracy:", accuracy_unaware)

# Fairness-Aware model using best k
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
            y_pred_unaware_sex = y_pred_unaware[np.isin(X_balanced[:, balanced_data_sex.columns.get_loc(sex_column)], [1])][:len(y_true)]
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

# Separate features and labels for the balanced data
X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

# K-Fold Cross Validation for Fairness-Unaware model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_values = range(1, 21)  # Test k values from 1 to 20
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

# Fairness-Unaware model using best k
model_unaware = KNeighborsClassifier(n_neighbors=best_k_unaware)
model_unaware.fit(X_balanced, y_balanced)
y_pred_unaware = model_unaware.predict(X_balanced)  # Using the balanced data for predictions
accuracy_unaware = accuracy_score(y_balanced, y_pred_unaware)

print("Fairness-Unaware Baseline Accuracy:", accuracy_unaware)

# Fairness-Aware model using best k
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
            y_pred_unaware_sex = y_pred_unaware[np.isin(X_balanced[:, balanced_data_sex.columns.get_loc(sex_column)], [1])][:len(y_true)]
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
    ax.set_title('Adult set, Undersampling')
    ax.set_xticks(x)
    ax.set_xticklabels(sex_labels)
    ax.legend()

    plt.show()
else:
    print("No valid accuracies to plot.")
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

    # Create bars for Fairness-Unaware and Fairness-Aware models with specified colors
    bars1 = ax.bar(x - width/2, [acc[0] for acc in accuracies_sex], width, label='Fairness-Unaware', color='blue')  # Male in blue
    bars2 = ax.bar(x + width/2, [acc[1] for acc in accuracies_sex], width, label='Fairness-Aware (Sex)', color='pink')  # Female in pink

    ax.set_ylabel('Accuracy')
    ax.set_title('Adult dataset, Undersampling')
    ax.set_xticks(x)
    ax.set_xticklabels(sex_labels)
    ax.legend()

    # Add the value labels on top of the bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.show()
else:
    print("No valid accuracies to plot.")
