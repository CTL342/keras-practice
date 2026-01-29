import pandas as pd

import keras
from keras import layers
from keras import ops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Preprocessing
df = pd.read_csv('cooling_tower_dataset.csv')
feature_cols = ['Outdoor Temp (°C)', 'Outdoor Humidity (%)', 'Wind Speed (m/s)', 'Water Inlet Temp (°C)', 'Water Flow Rate (L/s)']
target_col = 'Water Outlet Temp (°C)'

X = df[feature_cols].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Sequential Model
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear"),
])
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# Train Sequential Model
model.compile(optimizer='adam',
                loss = 'huber',
                metrics=['mae'])
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=500,
    batch_size=16,
    callbacks=[lr_scheduler],
    verbose=1
)

# Validation
loss, mae = model.evaluate(X_test_scaled, y_test)
sample_input = X_test_scaled[:5]
predictions = model.predict(sample_input)

print("\nComparing Real vs Predicted:")
for i in range(5):
    print(f"Real: {y_test[i]:.2f} | Predicted: {predictions[i][0]:.2f}")