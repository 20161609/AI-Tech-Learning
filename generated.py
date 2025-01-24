# Install necessary libraries and enable automatic timing for cells
!pip install --q ipython-autotime
%load_ext autotime

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras

# Load the Samsung stock dataset
samsung = pd.read_csv('/content/005930.KS.csv')
print(samsung.shape)  # Check the dataset dimensions

# Create a copy of the dataset for processing
df = samsung.copy()
df.head()  # Display the first few rows of the dataset

# Clean column names by replacing spaces with underscores and converting to lowercase
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
df.head()  # Display cleaned column names

# Check data information and types
df.info()

# Display dataset summary statistics
df.describe().T

# Check for rows with volume equal to 0
df[df['volume'] == 0]

# Replace volume equal to 0 with NaN and check missing values
df.loc[df['volume'] == 0, 'volume'] = np.nan
df.isna().sum()

# Drop rows with missing values
df = df.dropna()
df.isna().sum()  # Ensure no missing values remain

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.info()  # Verify the column type

# Set the 'date' column as the DataFrame index
df = df.set_index('date')
df.head()

# Plot closing prices and adjusted closing prices
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['close'], label='close')
plt.plot(df.index, df['adj_close'], label='adj_close')
plt.legend()
plt.show()

# Add moving averages (MA3 and MA5) columns
df['ma3'] = np.around(df['close'].rolling(window=3).mean(), 0)
df['ma5'] = np.around(df['close'].rolling(window=5).mean(), 0)
df.head()

# Calculate the mid-price between 'low' and 'high'
df['mid'] = (df['low'] + df['high']) / 2
df.head()

# Drop rows with missing values after adding new columns
df = df.dropna()
df.isna().sum()  # Verify no missing values remain

# Split the dataset into training (80%) and testing (20%) sets
idx = int(df.shape[0] * 0.8)
train = df.iloc[:idx, :]
test = df.iloc[idx:, :]
print(train.shape, test.shape)  # Check dimensions of the splits

# Prepare training data for the model
X_train = train.drop(['close', 'adj_close'], axis=1)
y_train = train['close']
print(X_train.shape, y_train.shape)

# Normalize features using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X_train_s = ms.fit_transform(X_train)
y_train = y_train.to_numpy()  # Convert target to NumPy array

# Create sequences for time series forecasting
def make_sequence_dataset(X, y, window_size):
    feature_list = []
    label_list = []

    for i in range(len(X) - window_size):
        feature_list.append(X[i:i+window_size])
        label_list.append(y[i+window_size])

    return np.array(feature_list), np.array(label_list)

# Generate sequences for training data
X_train_w, y_train_w = make_sequence_dataset(X_train_s, y_train, 20)
print(X_train_w.shape, y_train_w.shape)

# Build an LSTM model
from keras import layers
model = keras.Sequential()
model.add(layers.LSTM(32, activation='relu', input_shape=(20, 7)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

# Display model summary
model.summary()

# Compile the model
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mse', 'mae']
)

# Train the model
EPOCHS = 20
BATCH_SIZE = 16
history = model.fit(
    X_train_w, y_train_w,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# Function to plot training history
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(16, 8))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'], label='train loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='val loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot mean squared error (MSE) curve
    plt.subplot(1, 2, 2)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.plot(hist['epoch'], hist['mse'], label='train mse')
    plt.plot(hist['epoch'], hist['val_mse'], label='val mse')
    plt.title('MSE Curve')
    plt.legend()

    plt.show()

# Plot the training history
plot_history(history)

# Prepare testing data
X_test = test.drop(['close', 'adj_close'], axis=1)
y_test = test['close']

X_test_s = ms.transform(X_test)
y_test = y_test.to_numpy()

# Generate sequences for testing data
X_test_w, y_test_w = make_sequence_dataset(X_test_s, y_test, 20)

# Make predictions on testing data
y_pred = model.predict(X_test_w)

# Plot true vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test_w, label='true')
plt.plot(y_pred.flatten(), label='pred')
plt.legend()
plt.show()

# Multi-input LSTM model for further experimentation
# First input branch
input1 = layers.Input(shape=(20, 7))
x = layers.LSTM(64, activation='relu')(input1)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(132, activation='relu')(x)
output1 = layers.Dense(32, activation='relu')(x)

# Second input branch
input2 = layers.Input(shape=(20, 7))
x = layers.LSTM(64, activation='relu')(input2)
output2 = layers.Dense(32, activation='relu')(x)

# Merge outputs from both branches
merge = layers.Concatenate()([output1, output2])
output3 = layers.Dense(1)(merge)

# Define the final multi-input model
model = keras.Model(inputs=[input1, input2], outputs=output3)
model.summary()

# Visualize the model architecture
keras.utils.plot_model(model)

# Compile the multi-input model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Train the multi-input model
model.fit(
    [X_train_w, X_train_w], y_train_w,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# Make predictions with the multi-input model
y_pred = model.predict([X_test_w, X_test_w])

# Plot true vs predicted values for the multi-input model
plt.figure(figsize=(10, 5))
plt.plot(y_test_w, label='true')
plt.plot(y_pred.flatten(), label='pred')
plt.legend()
plt.show()
