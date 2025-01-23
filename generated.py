!pip install --q ipython-autotime
%load_ext autotime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from keras.datasets import fashion_mnist

data = fashion_mnist.load_data()

(X_train, y_train), (X_test, y_test) = data
X_train.shape, y_train.shape, X_test.shape, y_test.shape

columns = ['T-shirt/top', 'Trouser',  'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# np.random.seed(42)
sample = np.random.randint(60000, size=25)

fig = plt.figure(figsize=(8, 8))
for i, idx in enumerate(sample):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.axis('off')
    plt.title(columns[y_train[idx]])
fig.tight_layout()
plt.show()

pd.Series(y_train).value_counts().sort_index()

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape

X_train_s = X_train / 255.
X_val_s = X_val / 255.

X_train_s = X_train_s.reshape(-1, 28, 28, 1)
X_val_s = X_val_s.reshape(-1, 28, 28, 1)

X_train_s.shape, X_val_s.shape

y_train_o = to_categorical(y_train)
y_val_o = to_categorical(y_val)

y_train_o.shape, y_val_o.shape

from keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=3, activation='relu',
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.summary()

keras.utils.plot_model(model, dpi=72)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=['accuracy'])

EPOCHS = 30
BATCH_SIZE = 32

history = model.fit(X_train_s, y_train_o,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val_s, y_val_o))


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

X_test_s = X_test / 255.
X_test_s = X_test_s.reshape(-1, 28, 28, 1)
y_test_o = to_categorical(y_test)

y_pred = model.predict(X_test_s)

y_pred = np.argmax(y_pred, axis=1)
y_pred

# Define a function to print metrics and display the confusion matrix
def print_metrics(y_true, y_pred, aver='binary'):
    print('accuracy:', accuracy_score(y_true, y_pred))
    print('recall:', recall_score(y_true, y_pred, average=aver))
    print('precision:', precision_score(y_true, y_pred, average=aver))
    print('f1 :', f1_score(y_true, y_pred, average=aver))

    # Plot the confusion matrix
    cfm = confusion_matrix(y_true, y_pred)
    s = sns.heatmap(cfm, annot=True, cmap='Blues', fmt='d', cbar=False)
    s.set(xlabel='Prediction', ylabel='Actual')
    plt.show()

# Print metrics for the test dataset
print_metrics(y_test, y_pred, aver='macro')


model.save('cnn_muti_fashion.h5')

model.save('cnn_muti_fashion.keras')

# model.save('cnn_muti_fashion')