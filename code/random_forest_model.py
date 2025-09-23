import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib

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

bearing_data_shuffled = bearing_data.sample(frac=1)
bearing_data_shuffled = bearing_data_shuffled.astype(float)
print(bearing_data_shuffled.columns)

# target object Y, basically prediction target
y = bearing_data_shuffled.Y

# parameters X
params = ['DE_time', 'FE_time', 'DE_time (MA)',
          'FE_time (MA)', 'DE_time (SD)', 'FE_time (SD)']
X = bearing_data_shuffled[params]

# splitting the data into training & testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n----------TRAINING DATA----------")
print(X_train.shape, y_train.shape)

print("\n----------TEST DATA----------")
print(X_test.shape, y_test.shape)

# model specification (random forest)
bearing_model_rf = RandomForestRegressor(n_estimators=50, random_state=1, verbose=0)

# fitting model
bearing_model_rf.fit(X_train, y_train)  # train_X->feature, train_y->target

# predictions
val_pred_RF = bearing_model_rf.predict(X_test)

# calculating mae
val_mae_RF = mean_absolute_error(val_pred_RF, y_test)

print("\nRandom Forest MAE:", val_mae_RF)
print("Score = ", bearing_model_rf.score(X_test , y_test))
