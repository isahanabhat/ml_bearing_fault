import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib

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

bearing_data_shuffled = bearing_data.sample(frac=1)
bearing_data_shuffled = bearing_data_shuffled.astype(float)
print(bearing_data_shuffled.columns)

# target object Y, basically prediction target
y = bearing_data_shuffled.Y

# parameters X
params = ['DE_time', 'FE_time', 'DE_time (MA)',
          'FE_time (MA)', 'DE_time (SD)', 'FE_time (SD)']
X = bearing_data_shuffled[params]

# splitting the data into training & testing datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

print("\n----------TRAINING DATA----------")
print(X_train.shape, y_train.shape)

print("\n----------TEST DATA----------")
print(X_test.shape, y_test.shape)

print("\n----------VALIDATION DATA----------")
print(X_val.shape, y_val.shape)

# model specification
bearing_model_1 = XGBRegressor(n_estimators=500)
bearing_model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05, early_stopping_rounds=5)

# fitting w/out early stopping
bearing_model_1.fit(X_train, y_train)

# fitting w/ early stopping
bearing_model_2.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

print("\n----------PREDICTIONS 1----------")
pred_1 = bearing_model_1.predict(X_test)
print(pred_1)
print("MAE (bearing model 1) = ", mean_absolute_error(pred_1, y_test))
print("Score = ", bearing_model_1.score(X_test, y_test))

print("\n----------PREDICTIONS 2----------")
pred_2 = bearing_model_2.predict(X_test)
print(pred_2)
print("MAE (bearing model 2) = ", mean_absolute_error(pred_2, y_test))
print("Score = ", bearing_model_2.score(X_test, y_test))