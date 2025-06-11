
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load the German dataset (using sep='\s+' to avoid deprecation warning)
data = pd.read_csv('German/german.data', header=None, sep='\s+')

# Define column names (ensure this matches the dataset)
data.columns = ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 
                'employment', 'percent_of_income', 'other_parties', 'residence_since', 'property_magnitude', 
                'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 
                'class', 'credit_risk', 'age', 'personal_status']

# Inspect the unique values in the 'personal_status' column to understand gender encoding
print("Unique values in 'personal_status':")
print(data['personal_status'].unique())

# Map '1' to male (1) and '2' to female (0) assuming 1 = male, 2 = female in the dataset
data['sex'] = np.where(data['personal_status'] == 1, 1, 0)  # Adjust this if the mapping is different

# Check the distribution of male and female records
print("Gender distribution in the dataset before undersampling:")
print(data['sex'].value_counts())

# Preprocess the data: encoding categorical columns
data = pd.get_dummies(data, drop_first=True)

# Fairness-aware KNN classifier function for overall accuracy
def fairness_aware_knn(data, target_column, n_neighbors=5):
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_pred, y_test

# Calculate the accuracy on the original (imbalanced) dataset before undersampling
accuracy_original, predictions_original, y_test_original = fairness_aware_knn(data, target_column='class')
print(f"Overall Accuracy (before undersampling): {accuracy_original * 100:.2f}%")

# Separate the male and female data
data_male = data[data['sex'] == 1]  # Male data
data_female = data[data['sex'] == 0]  # Female data

# Male accuracy before undersampling
accuracy_male, _, y_test_male = fairness_aware_knn(data_male, target_column='class')
print(f"Accuracy for Males (before undersampling): {accuracy_male * 100:.2f}%")

# Female accuracy before undersampling
accuracy_female, _, y_test_female = fairness_aware_knn(data_female, target_column='class')
print(f"Accuracy for Females (before undersampling): {accuracy_female * 100:.2f}%")

# Fairness-aware KNN classifier function with undersampling
def fairness_aware_knn_with_undersampling(data, target_column, n_neighbors=5):
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into male and female data
    data_male = data[data['sex'] == 1]
    data_female = data[data['sex'] == 0]
    
    # Undersample the male data to match the number of female samples (300 males)
    data_male_undersampled = resample(data_male, 
                                      replace=False,    # Don't sample with replacement
                                      n_samples=300,    # Match the number of females
                                      random_state=42)  # Set seed for reproducibility
    
    # Combine the undersampled male data with the female data
    data_balanced = pd.concat([data_male_undersampled, data_female])
    
    # Print the number of males and females in the balanced dataset
    print(f"Number of males after undersampling: {data_male_undersampled.shape[0]}")
    print(f"Number of females in the dataset: {data_female.shape[0]}")
    
    # Split the balanced data into features (X) and target (y)
    X_balanced = data_balanced.drop(columns=[target_column])
    y_balanced = data_balanced[target_column]
    
    # Split the balanced data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)
    
    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_pred, y_test

# Calculate the accuracy on the balanced dataset after undersampling
accuracy_balanced, predictions_balanced, y_test_balanced = fairness_aware_knn_with_undersampling(data, target_column='class')
print(f"\nOverall Accuracy (after undersampling): {accuracy_balanced * 100:.2f}%")

# Separate the male and female data after undersampling
data_male_balanced = data[data['sex'] == 1]  # Male data
data_female_balanced = data[data['sex'] == 0]  # Female data

# Male accuracy after undersampling
accuracy_male_balanced, _, y_test_male_balanced = fairness_aware_knn_with_undersampling(data, target_column='class')
print(f"Accuracy for Males (after undersampling): {accuracy_male_balanced * 100:.2f}%")

# Female accuracy after undersampling
accuracy_female_balanced, _, y_test_female_balanced = fairness_aware_knn_with_undersampling(data, target_column='class')
print(f"Accuracy for Females (after undersampling): {accuracy_female_balanced * 100:.2f}%")
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# --- Adult Dataset Code ---
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

# Create a balanced dataset by upsampling the minority females
males = data[data['sex_Male'] == 1]
females = data[data['sex_Female'] == 1]

# Upsample the minority female class
females_upsampled = resample(females, 
                             replace=True,  # Sample with replacement
                             n_samples=len(males),  # Match the number of males
                             random_state=42)

# Combine the upsampled females with the males
balanced_data = pd.concat([males, females_upsampled])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels for both the original (unbalanced) and balanced data
X_original = data.drop(columns=['income']).values
y_original = data['income'].values

X_balanced = balanced_data.drop(columns=['income']).values
y_balanced = balanced_data['income'].values

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

# --- German Dataset Code ---
# Load the German dataset
data = pd.read_csv('German/german.data', header=None, sep='\s+')

# Define column names (ensure this matches the dataset)
data.columns = ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 
                'employment', 'percent_of_income', 'other_parties', 'residence_since', 'property_magnitude', 
                'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 
                'class', 'credit_risk', 'age', 'personal_status']

# Map '1' to male (1) and '2' to female (0) assuming 1 = male, 2 = female in the dataset
data['sex'] = np.where(data['personal_status'] == 1, 1, 0)  # Adjust this if the mapping is different

# Preprocess the data: encoding categorical columns
data = pd.get_dummies(data, drop_first=True)

# Fairness-aware KNN classifier function for overall accuracy
def fairness_aware_knn(data, target_column, n_neighbors=5):
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_pred, y_test

# Calculate the accuracy on the original (imbalanced) dataset before undersampling
accuracy_original, predictions_original, y_test_original = fairness_aware_knn(data, target_column='class')

# Separate the male and female data
data_male = data[data['sex'] == 1]  # Male data
data_female = data[data['sex'] == 0]  # Female data

# Male accuracy before undersampling
accuracy_male, _, y_test_male = fairness_aware_knn(data_male, target_column='class')

# Female accuracy before undersampling
accuracy_female, _, y_test_female = fairness_aware_knn(data_female, target_column='class')

# Fairness-aware KNN classifier function with undersampling
def fairness_aware_knn_with_undersampling(data, target_column, n_neighbors=5):
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into male and female data
    data_male = data[data['sex'] == 1]
    data_female = data[data['sex'] == 0]
    
    # Undersample the male data to match the number of female samples (300 males)
    data_male_undersampled = resample(data_male, 
                                      replace=False,    # Don't sample with replacement
                                      n_samples=300,    # Match the number of females
                                      random_state=42)  # Set seed for reproducibility
    
    # Combine the undersampled male data with the female data
    data_balanced = pd.concat([data_male_undersampled, data_female])
    
    # Split the balanced data into features (X) and target (y)
    X_balanced = data_balanced.drop(columns=[target_column])
    y_balanced = data_balanced[target_column]
    
    # Split the balanced data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)
    
    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_pred, y_test

# Calculate the accuracy on the balanced dataset after undersampling
accuracy_balanced, predictions_balanced, y_test_balanced = fairness_aware_knn_with_undersampling(data, target_column='class')

# Separate the male and female data after undersampling
data_male_balanced = data[data['sex'] == 1]  # Male data
data_female_balanced = data[data['sex'] == 0]  # Female data

# Male accuracy after undersampling
accuracy_male_balanced, _, y_test_male_balanced = fairness_aware_knn_with_undersampling(data, target_column='class')

# Female accuracy after undersampling
accuracy_female_balanced, _, y_test_female_balanced = fairness_aware_knn_with_undersampling(data, target_column='class')


# --- Plotting Code ---
# First, plot the Adult Dataset Accuracy Comparison
# Plotting the overall and gender-specific accuracies
x_labels_adult = ['Overall', 'Male', 'Female']
overall_accuracies_adult = [best_accuracy_unaware, accuracy_male, accuracy_female]
fairness_aware_accuracies_adult = [best_accuracy_fairness_aware, accuracy_male, accuracy_female]

# Create the plot for Adult Dataset
x_adult = np.arange(len(x_labels_adult))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1_adult = ax.bar(x_adult - width/2, overall_accuracies_adult, width, label='Fairness-Unaware')
bars2_adult = ax.bar(x_adult + width/2, fairness_aware_accuracies_adult, width, label='Fairness-Aware (Sex)')

# Add text for accuracy values inside bars in white
for bars in [bars1_adult, bars2_adult]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, f'{yval:.2f}', ha='center', va='bottom', color='white')

# Labeling and Titles
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison: Fairness-Unaware vs Fairness-Aware (Sex)')
ax.set_xticks(x_adult)
ax.set_xticklabels(x_labels_adult)
ax.legend()

# --- Plot the German Dataset Accuracy Comparison ---
x_labels_german = ['Overall', 'Male', 'Female']
overall_accuracies_german = [accuracy_original, accuracy_male, accuracy_female]
fairness_aware_accuracies_german = [accuracy_balanced, accuracy_male_balanced, accuracy_female_balanced]

# Create the plot for German Dataset
x_german = np.arange(len(x_labels_german))  # the label locations

fig, ax = plt.subplots(figsize=(10, 6))
bars1_german = ax.bar(x_german - width/2, overall_accuracies_german, width, label='Before Undersampling')
bars2_german = ax.bar(x_german + width/2, fairness_aware_accuracies_german, width, label='After Undersampling')

# Add text for accuracy values inside bars in white
for bars in [bars1_german, bars2_german]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, f'{yval:.2f}', ha='center', va='bottom', color='white')

# Labeling and Titles
ax.set_ylabel('Accuracy')
ax.set_title('German Set, Undersampling')
ax.set_xticks(x_german)
ax.set_xticklabels(x_labels_german)
ax.legend()

plt.tight_layout()
plt.show()
