# ================================================
# BEARING FAULT DIAGNOSIS - CNN + LSTM Hybrid
# Using Case Western Reserve University Dataset
# Author: Sahana Bhat (2025)
# Framework: TensorFlow / Keras
# ================================================

import numpy as np
import os
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. Load CWRU Dataset (.mat files)
# ------------------------------------------------
def load_cwru_data(data_dir):
    X = []
    y = []

    # Label map for basic fault categories
    label_map = {
        'Normal': 0, 'IR': 1, 'B': 2, 'OR': 3
    }

    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            data = loadmat(os.path.join(data_dir, filename))
            # Each .mat file usually has 'DE_time', 'FE_time', etc.
            key = [k for k in data.keys() if 'DE_time' in k][0]
            signal = data[key].ravel()

            # Segment the long vibration signal into shorter windows
            segment_length = 2048
            num_segments = len(signal) // segment_length
            signal = signal[:num_segments * segment_length]
            segments = signal.reshape(num_segments, segment_length)

            # Assign label from filename
            if 'Normal' in filename:
                label = label_map['Normal']
            elif 'IR' in filename:
                label = label_map['IR']
            elif 'B' in filename:
                label = label_map['B']
            elif 'OR' in filename:
                label = label_map['OR']
            else:
                continue

            X.append(segments)
            y += [label] * num_segments

    X = np.vstack(X)
    y = np.array(y)
    print(f"Loaded data: {X.shape[0]} samples, each {X.shape[1]} points")
    return X, y


# ------------------------------------------------
# 2. Preprocess Data
# ------------------------------------------------
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for CNN input: (samples, timesteps, channels)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------
# 3. Build CNN-LSTM Model
# ------------------------------------------------
def build_cnn_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(32, 7, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.LSTM(128, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------------------------------------------------
# 4. Train and Evaluate
# ------------------------------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")

    # Plot training results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend();
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend();
    plt.title("Loss")
    plt.show()

    return model


# ------------------------------------------------
# 5. Main Pipeline
# ------------------------------------------------
if __name__ == "__main__":
    # Set your dataset path here
    data_dir = "D:/datasets/CWRU"  # <-- change this to your CWRU folder

    # Step 1: Load Data
    X, y = load_cwru_data(data_dir)

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Model
    num_classes = len(np.unique(y))
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape, num_classes)
    model.summary()

    # Step 4: Train & Evaluate
    trained_model = train_and_evaluate(model, X_train, y_train, X_test, y_test)
