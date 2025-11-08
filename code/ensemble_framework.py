# =====================================================
# OPTION 1: COMBINE DE + FE DATA  (Fixed Residual CNN+BiLSTM)
# =====================================================

import scipy.io as sio
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM,
    BatchNormalization, Bidirectional, Input, LayerNormalization,
    Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import matplotlib.pyplot as plt


# -------------------------------
# 1️⃣ Load and Combine Data (DE + FE)
# -------------------------------
def load_mat_data(file_path):
    mat = sio.loadmat(file_path)
    key = [k for k in mat.keys() if 'DE' in k or 'FE' in k][0]
    return mat[key].squeeze()

base_dir = r"D:\SAHANABHAT\python_projects\ml_bearing_fault\mat_data\1797\de"

files = [
    "12k_DE_B007_0.mat", "12k_DE_IR007_0.mat", "12k_DE_OR007_3_0.mat",
    "12k_FE_B007_0.mat", "12k_FE_IR007_0.mat", "12k_FE_OR007_3_0.mat",
    "12k_FE_OR007_6_0.mat", "12k_FE_OR007_12_0.mat"
]

labels = [0, 1, 2, 0, 1, 2, 2, 2]  # Example: Ball, IR, OR

X, y = [], []
for f, label in zip(files, labels):
    data = load_mat_data(os.path.join(base_dir, f))
    segments = len(data) // 1024
    data = data[:segments * 1024].reshape(segments, 1024)
    X.append(data)
    y.extend([label] * segments)

X = np.vstack(X)
y = np.array(y)
print("✅ Combined Shape:", X.shape, y.shape)


# -------------------------------
# 2️⃣ Normalize and Split
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


# -------------------------------
# 3️⃣ Define Models
# -------------------------------

def build_simple_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 🧠 Fixed Residual CNN–BiLSTM
def build_residual_cnn_bilstm(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # --- Block 1 ---
    x = Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # --- Residual Block 1 (64→128 filters) ---
    shortcut = Conv1D(128, 1, padding='same')(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = Add()([x, shortcut])
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # --- Residual Block 2 (128→256 filters) ---
    shortcut = Conv1D(256, 1, padding='same')(x)
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = Add()([x, shortcut])
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)

    # --- BiLSTM Layers ---
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))(x)
    x = LayerNormalization()(x)

    # --- Dense Classifier ---
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    lr_schedule = CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=500)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# -------------------------------
# 4️⃣ Train Models
# -------------------------------
models = {
    "Simple_CNN": build_simple_cnn((1024, 1), len(set(y))),
    "CNN_LSTM": build_cnn_lstm((1024, 1), len(set(y))),
    "CNN_BiLSTM_Residual": build_residual_cnn_bilstm((1024, 1), len(set(y)))
}

histories = {}
epoch = [25, 60, 25]
i = 0
for name, model in models.items():
    print(f"\n🧩 Training model: {name}")
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epoch[i], batch_size=64,
        class_weight=class_weights,
        verbose=1, callbacks=[es, rl]
    )
    histories[name] = hist
    i += 1


# -------------------------------
# 5️⃣ Plot Accuracy & Loss
# -------------------------------
plt.figure(figsize=(12, 5))
for name, hist in histories.items():
    plt.plot(hist.history['accuracy'], label=f'{name} Train Acc')
    plt.plot(hist.history['val_accuracy'], '--', label=f'{name} Val Acc')
plt.title("📈 Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
for name, hist in histories.items():
    plt.plot(hist.history['loss'], label=f'{name} Train Loss')
    plt.plot(hist.history['val_loss'], '--', label=f'{name} Val Loss')
plt.title("📉 Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
