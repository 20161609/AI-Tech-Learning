!pip install --q ipython-autotime
%load_ext autotime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from io import BytesIO
import requests

data_url = 'https://raw.githubusercontent.com/20161609/data_box/main/mnist.npz'

# Download file from Url and load it to memory.
response = requests.get(data_url)
if response.status_code == 200:
    npz_data = BytesIO(response.content)  # Load it to memory
    df = np.load(npz_data)  # Read file on numpy

    # Check npz file's content
    print("Keys included in npz:", df.files)
    print("x_train shape:", df['x_train'].shape)
    print("y_train shape:", df['y_train'].shape)
else:
    print("Failed to download file:", response.status_code)

X_train = df['x_train']
X_test = df['x_test']
y_train = df['y_train']
y_test = df['y_test']

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# np.random.seed(42)
sample = np.random.randint(60000, size=25)
sample

fig = plt.figure(figsize=(8, 8))
for i, idx in enumerate(sample):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.axis('off')
    plt.title(y_train[idx])
fig.tight_layout()
plt.show()

sr = pd.Series(y_train).value_counts().sort_index()
sr

plt.bar(sr.index, sr)
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

## 3.학습용, 검증용 데이터 분리

X_train_s = X_train / 255.
X_val_s = X_val / 255.

y_train_o = to_categorical(y_train)
y_val_o = to_categorical(y_val)

y_train_o.shape, y_val_o.shape

X_train_s = X_train_s.reshape(-1, 28*28)
X_val_s = X_val_s.reshape(-1, 28*28)

X_train_s.shape, X_val_s.shape

# Define a neural network model
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),  # First hidden layer
    layers.Dense(32, activation='relu'),                     # Second hidden layer
    layers.Dense(16, activation='relu'),                     # Third hidden layer
    layers.Dense(10, activation='softmax'),                  # Output layer for 10 classes
])

# Display the model summary
model.summary()

# Compile the model with Adam optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    optimizer=adam,
    metrics=['accuracy']             # Metric to monitor during training
)

# Set training parameters
EPOCHS = 30
BATCH_SIZE = 32

# Train the model with training data and validate on validation data
history = model.fit(
    X_train_s, y_train_o,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_s, y_val_o)
)

# Function to plot training history
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(16, 8))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()

# Plot training history
plot_history(history)

X_test_s = X_test / 255.
X_test_s = X_test_s.reshape(-1, 28*28)  # Flatten images
y_test_o = to_categorical(y_test)       # One-hot encode test labels

X_test_s.shape, y_test_o.shape

# Make predictions on test data
y_pred = model.predict(X_test_s)

# Convert predictions from probabilities to class labels
y_pred = np.argmax(y_pred, axis=1)

def print_metrics(y_true, y_pred, aver='binary'):
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred, average=aver))
    print('Precision:', precision_score(y_true, y_pred, average=aver))
    print('F1 Score:', f1_score(y_true, y_pred, average=aver))

    # Plot confusion matrix
    cfm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cfm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

print_metrics(y_test, y_pred, aver='macro')

sample = np.random.randint(10000, size=25)

fig = plt.figure(figsize=(8, 8))
for i, idx in enumerate(sample):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_test[idx], cmap='gray')  # Display grayscale image
    plt.axis('off')
    plt.title(f'Pred: {y_pred[idx]} (True: {y_test[idx]})')  # Show predicted and true labels
fig.tight_layout()
plt.show()