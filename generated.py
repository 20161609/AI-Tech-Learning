import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from keras.utils import to_categorical


SEED = 42
TARGET = 'species'

url_iris = 'https://raw.githubusercontent.com/20161609/data_box/refs/heads/main/iris.csv'
df = pd.read_csv(url_iris)
df.shape

print('Before:',list(df.columns))

col_dict = {col: col.lower().replace(' ', '_' ) for col in df.columns}
df.rename(columns=col_dict, inplace=True)

print('After:',list(df.columns))

# Handle missing values in numeric columns by filling with mean
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        print(f"Filling missing values in numeric column '{col}' with mean.")
        df[col].fillna(df[col].mean(), inplace=True)

# Handle missing values in categorical columns by filling with mode
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
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

train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df[TARGET])

train.shape, test.shape

X_train = train.drop(TARGET, axis=1)
y_train = train[TARGET]
X_test = test.drop(TARGET, axis=1)
y_test = test[TARGET]

X_train.shape, y_train.shape, X_test.shape, y_test.shape

rs = RobustScaler()
X_train_s = rs.fit_transform(X_train)
X_test_s = rs.transform(test.drop(TARGET, axis=1))

X_train_s.shape

y_train = to_categorical(y_train, num_classes=3)

from keras import layers

input_shape = X_train_s.shape[1:]

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=input_shape),
    layers.Dense(8, activation='relu'),
    layers.Dense(3, activation='softmax')
    # eng -> Sigmoid when you executes the multi-classification
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

EPOCH = 100
BATCH_SIZE = 16
history = model.fit(
    X_train_s, y_train,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.plot(hist['epoch'], hist['loss'], label='train loss')
  plt.plot(hist['epoch'], hist['val_loss'], label='val loss')
  plt.title('Loss Curve')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.plot(hist['epoch'], hist['accuracy'], label='train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'], label='val accuracy')
  plt.title('Accuracy Curve')
  plt.legend()
  plt.show()

plot_history(history)

y_pred = model.predict(X_test_s)
y_pred.shape

import numpy as np

y_pred = np.argmax(y_pred, axis=1)
y_pred

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix

def print_metrics(y_true, y_pred, ave='binary'):
  print('accuracy:', accuracy_score(y_test, y_pred))
  print('recall:', recall_score(y_test, y_pred, average=ave))
  print('precision:', precision_score(y_test, y_pred, average=ave))
  print('f1 :', f1_score(y_test, y_pred, average=ave))

  clm = confusion_matrix(y_test, y_pred)
  s = sns.heatmap(clm, annot=True, cmap='Blues', fmt='d', cbar=False)
  s.set(xlabel='Predicted', ylabel='Actual')

print_metrics(y_test, y_pred, ave='macro')