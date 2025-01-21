import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Versions of TensorFlow and Keras
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([9, 12, 15, 18, 21, 24, 27, 30, 33, 36])


plt.scatter(X, y)
plt.title("Training Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

plt.scatter(X_train, y_train, label="Training Data")
plt.scatter(X_test, y_test, label="Test Data", color="orange")
plt.title("Training and Test Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Reshape the data for training
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(f"Linear Regression Coef: {lr.coef_}, Intercept: {lr.intercept_}")

# Predictions and evaluation
lr_y_pred = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)

print(f"Linear Regression Coef: {lr.coef_}, Intercept: {lr.intercept_}")
print(f"Linear Regression Test MSE: {lr_mse}, MAE: {lr_mae}")

# Neural Network
model = Sequential([
    layers.Input(shape=(1,)),  # Input layer
    layers.Dense(units=1)     # Dense layer
])

model.summary()

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mse']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=len(X_train),
    verbose=0
)


# Evaluate the model
nn_eval = model.evaluate(X_test, y_test, verbose=0)
nn_mse = nn_eval[0]
print(f"Neural Network Test MSE: {nn_mse}")


# Predictions
y_pred = model.predict(X_test)



# Compare predictions
comparison = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Linear Regression Prediction": lr_y_pred.flatten(),
    "Neural Network Prediction": y_pred.flatten()
})
print(comparison)

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training History")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot predictions
plt.scatter(X_test, y_test, label="Test Data")
plt.plot(X_test, lr_y_pred, 'r', label="Linear Regression Fit")
plt.plot(X_test, y_pred, 'g', label="Neural Network Fit")
plt.title("Model Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
