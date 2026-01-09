import os

# Fix for WinError 2 / joblib on Python 3.13
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_and_preprocess(file_paths):
    records = []
    all_macs = set()

    # Pass 1: Identify all unique Wi-Fi MAC addresses
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found.")
            continue
        with open(path, 'r') as f:
            data = json.load(f)
            for key, val in data.items():
                if key.startswith('collection'):
                    for fp in val.get('fingerprints', []):
                        wifi = fp.get('wifi', {})
                        all_macs.update(wifi.keys())

    mac_list = sorted(list(all_macs))
    print(f"Unique Wi-Fi Access Points found: {len(mac_list)}")

    # Pass 2: Build the feature vectors
    for path in file_paths:
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            data = json.load(f)
            for key, val in data.items():
                if key.startswith('collection'):
                    x, y, z = val['x'], val['y'], val['z']
                    for fp in val.get('fingerprints', []):
                        wifi = fp.get('wifi', {})
                        vector = []
                        for mac in mac_list:
                            if mac in wifi:
                                rssis = wifi[mac].get('rssi', [])
                                # Clean data: ensure it's a list and not empty
                                if isinstance(rssis, list) and len(rssis) > 0:
                                    vector.append(np.mean(rssis))
                                else:
                                    vector.append(-105)  # "Silent" threshold
                            else:
                                vector.append(-105)  # "Silent" threshold

                        records.append({'coords': [x, y, z], 'features': vector})

    return np.array([r['features'] for r in records]), \
        np.array([r['coords'] for r in records]), \
        mac_list


# --- MAIN EXECUTION ---

# Define your files
data_files = ['0-pixel-04-03-2021_15-32-38.json', '0-redmi-04-03-2021_15-31-12.json', '0-pixel-04-06-2021_19-34-41.json', '0-redmi-04-06-2021_19-33-47.json',
              '1-pixel-04-06-2021_21-09-29.json', '1-pixel-25-02-2021_21-16-46.json', '1-redmi-04-06-2021_21-09-31.json', '1-redmi-25-02-2021_21-16-10.json']

# 1. Load Data
X, y, mac_list = load_and_preprocess(data_files)

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a Pipeline: Scale the data, then apply KNN
# Scaling helps the model compare signal strengths more accurately
model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nTraining Complete.")
print(f"Average Localization Error (RMSE): {rmse:.2f} meters")

# 6. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the test set results
ax.scatter(y_test[:, 0], y_test[:, 1], y_test[:, 2], c='blue', label='Ground Truth (Target)', alpha=0.6)
ax.scatter(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], c='red', marker='x', label='Predicted (Estimated)', alpha=0.9)

# Draw error lines
for i in range(len(y_test)):
    ax.plot([y_test[i, 0], y_pred[i, 0]],
            [y_test[i, 1], y_pred[i, 1]],
            [y_test[i, 2], y_pred[i, 2]], 'black', linestyle='--', alpha=0.2)

ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.set_zlabel('Z Coordinate (m)')
ax.set_title('Wi-Fi Fingerprinting Localization Results')
ax.legend()

plt.show()