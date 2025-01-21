import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

SEED = 42

data_url = 'https://raw.githubusercontent.com/20161609/data_box/refs/heads/main/penguins.csv'
df = pd.read_csv(data_url)
df.shape

df.head()

def clean_column_names(col):
  # Change to lowercase and remove spaces and special characters ('_', '(', ')')
  col = col.strip()
  col = col.lower()
  col = col.replace(' ', '_')
  col = col.replace('(', '')
  col = col.replace(')', '')
  return col

df.columns = [clean_column_names(col) for col in df.columns]
df.head()

df.info()

# Handle missing values in numeric columns by filling with mean
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        print(f"Filling missing values in numeric column '{col}' with mean.")
        df[col].fillna(df[col].mean(), inplace=True)

# Handle missing values in categorical columns by filling with mode
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['number']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        print(f"Filling missing values in categorical column '{col}' with mode.")
        df[col].fillna(df[col].mode()[0], inplace=True)


# Convert categorical columns to numerical using Label Encoding
for col in categorical_cols:
  print(f"Encoding categorical column '{col}'.")

  le = LabelEncoder()
  # Convert to string before encoding
  df[col] = le.fit_transform(df[col].astype(str))

print("Missing values after preprocessing:")
print(df.isnull().sum())

initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
final_rows = df.shape[0]
print(f"Removed {initial_rows - final_rows} duplicate rows.")

df.describe().T

cols_num = df[numerical_cols]
cols_num

cols_num.hist(figsize=(10, 8))
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()


for i, col in enumerate(cols_num.columns):
  sns.boxplot(y=col, data=cols_num, hue=df['species'], ax=axes[i])
  axes[i].set_xlabel(None)
  axes[i].set_ylabel(None)
  axes[i].set_title(col)

# Separate features (X) and target (y)
target_col = 'species'  # Assuming 'species' is the target column
X = df.drop(columns=[target_col])
y = df[target_col]


# sns.heatmap(train.isna())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=SEED)
dt_model.fit(X_train_scaled, y_train)
y_pred_tree = dt_model.predict(X_test_scaled)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=SEED)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the models
print("Decision Tree Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tree, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_tree, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_tree, average='weighted'):.4f}")

print("\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")


# Get class names as strings
class_names = le.inverse_transform(dt_model.classes_)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=class_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()