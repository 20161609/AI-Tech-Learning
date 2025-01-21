import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_link = "https://raw.githubusercontent.com/20161609/data_box/refs/heads/main/diabetes.csv"
diabetes = pd.read_csv(data_link)

# Check the dataset
print("Dataset Shape:", diabetes.shape)
print(diabetes.head())

# Handle missing values
if diabetes.isnull().sum().any():
    print("Missing values detected. Filling missing values with mean.")
    diabetes.fillna(diabetes.mean(), inplace=True)

# Remove duplicate rows
initial_rows = diabetes.shape[0]
diabetes.drop_duplicates(inplace=True)
final_rows = diabetes.shape[0]
print(f"Removed {initial_rows - final_rows} duplicate rows.")

# Split the data into features and labels
X = diabetes.iloc[:, :-1]  # Exclude the last column (features)
y = diabetes.iloc[:, -1]   # Select the last column (label)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the initial model
y_pred = svm_model.predict(X_test)
print("Initial Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optimize the model using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the optimized model
optimized_model = grid_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test)

print("Optimized Model Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))
