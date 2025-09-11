import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation

# encoding utf-8
import os
import sys
import io
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# data import
file_healthy = r"../csv_data/1797/normal_0_1797.csv"
file_faulty = r"../csv_data/1797/12k_de_IR007_0.csv"

bearing_data_h = pd.read_csv(file_healthy)
bearing_data_h.dropna(inplace=True)
bearing_data_h = bearing_data_h.drop(0)

bearing_data_f = pd.read_csv(file_faulty)
bearing_data_f.dropna(inplace=True)
bearing_data_f = bearing_data_f.drop(0)
bearing_data_f.drop('BA_time', axis=1, inplace=True)

bearing_data = pd.concat([bearing_data_h, bearing_data_f])
bearing_data.columns = bearing_data.columns.str.encode('ascii', 'ignore').str.decode('ascii')
print(bearing_data.shape)

# file = r"..\csv_data\combined_1797_12k.csv"
# bearing_data = pd.read_csv(file)
bearing_data_shuffled = bearing_data.sample(frac=1)
print(bearing_data_shuffled.columns)

# feature X
params = ['FE_time', 'DE_time', 'DE_time (MA)', 'FE_time (MA)', 'DE_time (SD)', 'FE_time (SD)']
X = bearing_data_shuffled[params]

# target object Y, basically prediction target
y = bearing_data_shuffled.Y

# normalization
scaler_std = StandardScaler()
#X.values[:] = scaler_std.fit_transform(X)
X = scaler_std.fit_transform(X)

scaler_minmax = MinMaxScaler()
#X.values[:] = scaler_minmax.fit_transform(X)
X = scaler_minmax.fit_transform(X)

# splitting of data into train, test, val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
l2_reg = tensorflow.keras.regularizers.L2(l2=0.01)

#defining the model
model.add(Flatten())
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
)

history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        batch_size=32,
        verbose = 1
)

print(model.summary())

# model evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy = ", round(accuracy*100, 2))

y_predicted_by_model = model.predict(X_test)

# plot accuracy and loss
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist = hist[1:]

plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["font.size"] = 9
plt.rcParams["font.weight"] = "bold"
fig, axs = plt.subplots(nrows = 1, ncols = 2)

axs[0].plot(hist['epoch'], hist['loss'], label="loss", color='red')
axs[0].plot(hist['epoch'], hist['val_loss'],label=" val_loss", color='blue')
axs[0].set_title('Loss'); axs[0].grid(True)
axs[0].legend()

axs[1].plot(hist['epoch'], hist['accuracy'], label="accuracy", color='red')
axs[1].plot(hist['epoch'], hist['val_accuracy'], label="val_accuracy", color='blue')
axs[1].set_title('Accuracy'); axs[1].grid(True)
axs[1].legend()
fig.tight_layout(); plt.show()