'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
print("\nGender distribution in the dataset (before SMOTE):")
print(data['sex'].value_counts())

# Preprocess the data: encoding categorical columns
data = pd.get_dummies(data, drop_first=True)

# Fairness-unaware KNN classifier function
def fairness_unaware_knn(data, target_column, n_neighbors=5):
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

# **Before Oversampling**

# Overall accuracy before oversampling
accuracy, predictions, y_test = fairness_unaware_knn(data, target_column='class')
print(f"\nOverall Accuracy before Oversampling: {accuracy * 100:.2f}%")

# Filter by male and female
data_male = data[data['sex'] == 1]  # Male data
data_female = data[data['sex'] == 0]  # Female data

# Print the number of male and female samples before oversampling
print(f"Number of Male records before Oversampling: {len(data_male)}")
print(f"Number of Female records before Oversampling: {len(data_female)}")

# Calculate accuracy for males before oversampling
accuracy_male, _, _ = fairness_unaware_knn(data_male, target_column='class')
print(f"Accuracy for Males before Oversampling: {accuracy_male * 100:.2f}%")

# Calculate accuracy for females before oversampling
accuracy_female, _, _ = fairness_unaware_knn(data_female, target_column='class')
print(f"Accuracy for Females before Oversampling: {accuracy_female * 100:.2f}%")

# **Calculate required female samples to reach 700 (currently 300)**

# Calculate the number of samples needed for female class
samples_needed = 700 - len(data_female)  # 700 females required
print(f"Samples needed to reach 700 females: {samples_needed}")

# Randomly replicate the female samples to reach 700
data_female_upsampled = data_female.sample(n=700, replace=True, random_state=42)

# Combine male data and upsampled female data
data_resampled = pd.concat([data_male, data_female_upsampled], axis=0)

# **After Oversampling**

# Run KNN on the resampled dataset and use 'class' as the target column
accuracy_resampled, predictions_resampled, y_test_resampled = fairness_unaware_knn(data_resampled, target_column='class')
print(f"\nOverall Accuracy after Oversampling: {accuracy_resampled * 100:.2f}%")

# Filter by male and female in the resampled data
data_male_resampled = data_resampled[data_resampled['sex'] == 1]  # Male data
data_female_resampled = data_resampled[data_resampled['sex'] == 0]  # Female data

# Print the number of male and female samples after oversampling
print(f"Number of Male records after Oversampling: {len(data_male_resampled)}")
print(f"Number of Female records after Oversampling: {len(data_female_resampled)}")

# Ensure there are enough records for both males and females
if len(data_male_resampled) > 0 and len(data_female_resampled) > 0:
    # Evaluate male accuracy after oversampling
    accuracy_male_resampled, _, y_test_male_resampled = fairness_unaware_knn(data_male_resampled, target_column='class')
    # Evaluate female accuracy after oversampling
    accuracy_female_resampled, _, y_test_female_resampled = fairness_unaware_knn(data_female_resampled, target_column='class')

    # Print male and female accuracy after oversampling
    print(f"Accuracy for Males after Oversampling: {accuracy_male_resampled * 100:.2f}%")
    print(f"Accuracy for Females after Oversampling: {accuracy_female_resampled * 100:.2f}%")
else:
    print("Insufficient data for one or both gender groups after oversampling.")
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
print("\nGender distribution in the dataset (before SMOTE):")
print(data['sex'].value_counts())

# Preprocess the data: encoding categorical columns
data = pd.get_dummies(data, drop_first=True)

# Fairness-unaware KNN classifier function
def fairness_unaware_knn(data, target_column, n_neighbors=5):
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

# **Before Oversampling**

# Overall accuracy before oversampling
accuracy, predictions, y_test = fairness_unaware_knn(data, target_column='class')
print(f"\nOverall Accuracy before Oversampling: {accuracy * 100:.2f}%")

# Filter by male and female
data_male = data[data['sex'] == 1]  # Male data
data_female = data[data['sex'] == 0]  # Female data

# Print the number of male and female samples before oversampling
print(f"Number of Male records before Oversampling: {len(data_male)}")
print(f"Number of Female records before Oversampling: {len(data_female)}")

# Calculate accuracy for males before oversampling
accuracy_male, _, _ = fairness_unaware_knn(data_male, target_column='class')
print(f"Accuracy for Males before Oversampling: {accuracy_male * 100:.2f}%")

# Calculate accuracy for females before oversampling
accuracy_female, _, _ = fairness_unaware_knn(data_female, target_column='class')
print(f"Accuracy for Females before Oversampling: {accuracy_female * 100:.2f}%")

# **Calculate required female samples to reach 700 (currently 300)**

# Calculate the number of samples needed for female class
samples_needed = 700 - len(data_female)  # 700 females required
print(f"Samples needed to reach 700 females: {samples_needed}")

# Randomly replicate the female samples to reach 700
data_female_upsampled = data_female.sample(n=700, replace=True, random_state=42)

# Combine male data and upsampled female data
data_resampled = pd.concat([data_male, data_female_upsampled], axis=0)

# **After Oversampling**

# Run KNN on the resampled dataset and use 'class' as the target column
accuracy_resampled, predictions_resampled, y_test_resampled = fairness_unaware_knn(data_resampled, target_column='class')
print(f"\nOverall Accuracy after Oversampling: {accuracy_resampled * 100:.2f}%")

# Filter by male and female in the resampled data
data_male_resampled = data_resampled[data_resampled['sex'] == 1]  # Male data
data_female_resampled = data_resampled[data_resampled['sex'] == 0]  # Female data

# Print the number of male and female samples after oversampling
print(f"Number of Male records after Oversampling: {len(data_male_resampled)}")
print(f"Number of Female records after Oversampling: {len(data_female_resampled)}")

# Ensure there are enough records for both males and females
if len(data_male_resampled) > 0 and len(data_female_resampled) > 0:
    # Evaluate male accuracy after oversampling
    accuracy_male_resampled, _, y_test_male_resampled = fairness_unaware_knn(data_male_resampled, target_column='class')
    # Evaluate female accuracy after oversampling
    accuracy_female_resampled, _, y_test_female_resampled = fairness_unaware_knn(data_female_resampled, target_column='class')

    # Print male and female accuracy after oversampling
    print(f"Accuracy for Males after Oversampling: {accuracy_male_resampled * 100:.2f}%")
    print(f"Accuracy for Females after Oversampling: {accuracy_female_resampled * 100:.2f}%")
else:
    print("Insufficient data for one or both gender groups after oversampling.")

# Plotting the accuracy comparison

# Define the accuracy values for plotting
accuracies = {
    'Before Oversampling': {
        'Male': accuracy_male * 100,
        'Female': accuracy_female * 100
    },
    'After Oversampling': {
        'Male': accuracy_male_resampled * 100,
        'Female': accuracy_female_resampled * 100
    }
}

# Plotting
labels = ['Male', 'Female']
before_accuracies = [accuracies['Before Oversampling']['Male'], accuracies['Before Oversampling']['Female']]
after_accuracies = [accuracies['After Oversampling']['Male'], accuracies['After Oversampling']['Female']]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Bar positions for before and after oversampling
x = np.arange(len(labels))  # x locations for the labels
width = 0.35  # the width of the bars

# Create bars for before and after oversampling
bars_before = ax.bar(x - width/2, before_accuracies, width, label='Before Oversampling', color='blue')
bars_after = ax.bar(x + width/2, after_accuracies, width, label='After Oversampling', color='orange')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Accuracy (%)')
ax.set_title('German Set, Oversampling')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the values on top of the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add the labels to the bars
add_value_labels(bars_before)
add_value_labels(bars_after)

# Show the plot
plt.tight_layout()
plt.show()
