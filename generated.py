import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the dataset for KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a range of k values to test
k_values = range(1, 21)
accuracies = []

# Train KNN for different k values and evaluate accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Create a DataFrame to visualize the accuracy results
results_df = pd.DataFrame({'k': k_values, 'Accuracy': accuracies})

# Display the results to the user
import ace_tools as tools; 
tools.display_dataframe_to_user(name="KNN Accuracy Results", dataframe=results_df)

# Plot the accuracy as a function of k
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()
