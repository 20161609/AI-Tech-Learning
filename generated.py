!pip install ipython-autotime
%load_ext autotime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers


link = 'https://raw.githubusercontent.com/20161609/data_box/refs/heads/main/auto-mpg.csv'
df = pd.read_csv(link)

df.shape

df.head()

df.info()

df.columns = [col.replace(' ', '_') for col in df.columns]
df.head()

df['horsepower'].value_counts()

# Identify and remove rows with missing values (indicated by '?') in 'horsepower'

print(f"A row containing {len(df[df['horsepower']=='?'])} question marks was found.")
df = df[df['horsepower']!='?']

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

df = df.drop(['origin','car_name'], axis=1)
df.head()

# Split the dataset into training and testing sets (80-20 split)
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.shape, test.shape

# Separate features (X) and target variable (y) for training
X_train = train.drop('mpg', axis=1)
y_train = train['mpg']
X_test = test.drop('mpg', axis=1)
y_test = test['mpg']

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Scale the feature data using StandardScaler for normalization
ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)
X_train_s

# Print the types and shapes of the scaled feature set and target variable
X_train_s.shape

# Scale the test features using the same scaler fitted on the training data
X_test_s = ss.transform(X_test)
X_test_s

# Convert the target variable of the test set to a NumPy array
y_test = y_test.to_numpy()
y_test

X_test_s.shape, y_test.shape



model = keras.Sequential([
    layers.Dense(units=5, activation='relu', input_shape=(6,)),  # Input layer with 6 features
    layers.Dense(units=3, activation='relu'),                   # Hidden layer with 3 neurons
    layers.Dense(units=1)                                      # Output layer for regression
])

model.summary()


# Compile the model with Mean Squared Error (MSE) loss, Adam optimizer, and evaluation metrics
model.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mse","mae"]
)

# Define training parameters
EPOCHS = 100
BATCH_SIZE = 16

# Train the model on the training data, using 20% of it for validation
history = model.fit(
    X_train_s, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# Extract the training history for analysis
hist = history.history
epochs = history.epoch

# Plot the training and validation loss curves for visualization
plt.plot(epochs, hist['loss'], label='train')
plt.plot(epochs, hist['val_loss'], label='val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Predict the target variable (mpg) using the trained model
y_pred = model.predict(X_test_s)

# Calculate evaluation metrics (Root Mean Squared Error and Mean Absolute Error)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE:', mae)

# Visualize the predictions against actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.plot([0, 50], [0, 50], 'r')  # Reference line for perfect predictions
plt.show()