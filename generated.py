import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42

data_link = "https://raw.githubusercontent.com/20161609/data_box/refs/heads/main/diabetes.csv"
diabetes = pd.read_csv(data_link)

print("Dataset Shape:", diabetes.shape)

diabetes.head()

diabetes.describe().T

if diabetes.isnull().sum().any():
    print("Missing values detected. Filling missing values with mean.")
    diabetes.fillna(diabetes.mean(), inplace=True)

initial_rows = diabetes.shape[0]
diabetes.drop_duplicates(inplace=True)
final_rows = diabetes.shape[0]
print(f"Removed {initial_rows - final_rows} duplicate rows.")

# Identify categorical columns and encode them using Label Encoding
categorical_cols = diabetes.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", categorical_cols)

# Encoding
for col in categorical_cols:
    if diabetes[col].isnull().sum() > 0:
        print(f"Filling missing values in '{col}' with mode.")
        diabetes[col].fillna(diabetes[col].mode()[0], inplace=True)

X = diabetes.iloc[:, :-1]  # features
y = diabetes.iloc[:, -1]   # label

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Define a function to print classification metrics and display a confusion matrix heatmap
def print_metrics(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    # Plt show
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

print_metrics(y_test, y_pred)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print("Best Parameters:", grid_search.best_params_)

optimized_model = grid_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test)

print("Optimized Model Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))

print_metrics(y_test, y_pred_optimized)