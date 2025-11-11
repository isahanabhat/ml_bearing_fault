import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_cwru_data(data_dir, segment_length=2048, test_size=0.2, random_state=42):
    """
    Load and preprocess CWRU bearing dataset (.mat files).
    Each file is segmented and labeled automatically.
    """
    label_map = {
        'normal': 0,
        'B': 1,   # Ball fault
        'IR': 2,  # Inner race
        'OR': 3   # Outer race
    }

    X, y = [], []

    for file in os.listdir(data_dir):
        if not file.endswith('.mat'):
            continue

        path = os.path.join(data_dir, file)
        data = loadmat(path)

        # Find the Drive End key (commonly contains "DE")
        key = [k for k in data.keys() if 'DE' in k][0]
        signal = data[key].ravel()

        # Normalize signal
        signal = (signal - np.mean(signal)) / np.std(signal)

        # Segment the signal
        n_segments = len(signal) // segment_length
        signal = signal[:n_segments * segment_length]
        segments = signal.reshape(n_segments, segment_length)

        # Assign label based on filename
        if "normal" in file.lower():
            label = label_map['normal']
        elif "B" in file:
            label = label_map['B']
        elif "IR" in file:
            label = label_map['IR']
        elif "OR" in file:
            label = label_map['OR']
        else:
            continue

        X.append(segments)
        y += [label] * n_segments

    # Combine all signals and labels
    X = np.vstack(X)
    y = np.array(y)

    # Reshape for CNN input
    X = X[..., np.newaxis]

    # Standardize across dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Loaded CWRU Dataset from {data_dir}")
    print(f"   Total samples: {X.shape[0]} | Segment length: {segment_length}")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test
