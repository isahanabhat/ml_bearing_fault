import scipy
import pandas as pd

h_1797_12k_normal = r"../mat_data/1797/normal_0_1797.mat"
f_1797_12k_de_b007 = r"../mat_data/1797/de/12k_de_B007_0.mat"
f_1797_12k_de_ir007 = r"../mat_data/1797/de/12k_de_IR007_0.mat"
f_1797_12k_de_or007_3 = r"../mat_data/1797/de/12k_de_OR007_3_0.mat"
f_1797_12k_de_or007_6 = r"../mat_data/1797/de/12k_de_OR007_6_0.mat"
f_1797_12k_de_or007_12 = r"../mat_data/1797/de/12k_de_OR007_12_0.mat"

h_1797_12k_normal_csv = r"../csv_data/1797/normal_0_1797.csv"
f_1797_12k_de_b007_csv = r"../csv_data/1797/12k_de_B007_0.csv"
f_1797_12k_de_ir007_csv = r"../csv_data/1797/12k_de_IR007_0.csv"
f_1797_12k_de_or007_3_csv = r"../csv_data/1797/12k_de_OR007_3_0.csv"
f_1797_12k_de_or007_6_csv = r"../csv_data/1797/12k_de_OR007_6_0.csv"
f_1797_12k_de_or007_12_csv = r"../csv_data/1797/12k_de_OR007_12_0.csv"


source_files = [[h_1797_12k_normal, 0], [f_1797_12k_de_ir007, 1], [f_1797_12k_de_b007, 2],
                [f_1797_12k_de_or007_3, 3] , [f_1797_12k_de_or007_6, 4] , [f_1797_12k_de_or007_12, 5]]
dest_files = [h_1797_12k_normal_csv, f_1797_12k_de_ir007_csv, f_1797_12k_de_b007_csv,
              f_1797_12k_de_or007_3_csv, f_1797_12k_de_or007_6_csv, f_1797_12k_de_or007_12_csv]

faulty_combined = r"../big_data/faulty_combined_1797_12k.csv"
final_combined = r"../big_data/combined_1797_12k.csv"

# de -> drive end accelerometer data
# fe -> fan end accelerometer data

datalist = []

def mat_to_csv(source, destination, status):
    dataset = scipy.io.loadmat(source)
    # print(type(dataset))
    header = list(dataset.keys())
    # print(header)

    col = []
    for i in header[::-1]:
        if i == '__globals__':
            break
        col.append(i)

    df = pd.DataFrame()
    for i in col[1:]:
        col_name = i.split("_", 1)[1]
        df[col_name] = dataset[i].tolist()
    df['Y'] = status

    df = df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    df["DE_time (MA)"] = df["DE_time"].rolling(window=5).mean()
    df["FE_time (MA)"] = df["FE_time"].rolling(window=5).mean()
    df["DE_time (SD)"] = df["DE_time"].rolling(window=5).std()
    df["FE_time (SD)"] = df["FE_time"].rolling(window=5).std()

    df = df.fillna(0)

    datalist.append(df)
    df.to_csv(destination, index=False)
    print("Succesful!")

mat_to_csv(source_files[0][0], dest_files[0], source_files[0][1])

for i in range(1,6):
    mat_to_csv(source_files[i][0], dest_files[i], source_files[i][1])

faulty_data = datalist[1:]
faulty_df = pd.concat(faulty_data, axis=0)

faulty_df.to_csv(faulty_combined, index=False)

final_dataframe = pd.concat(datalist, axis=0)
final_dataframe = final_dataframe.drop(0)

final_dataframe = final_dataframe.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

# adding "Moving Avg." & SD
final_dataframe["DE_time (MA)"] = final_dataframe["DE_time"].rolling(window=5).mean()
final_dataframe["FE_time (MA)"] = final_dataframe["FE_time"].rolling(window=5).mean()
final_dataframe["DE_time (SD)"] = final_dataframe["DE_time"].rolling(window=5).std()
final_dataframe["FE_time (SD)"] = final_dataframe["FE_time"].rolling(window=5).std()

final_dataframe = final_dataframe.fillna(0)
# print(final_dataframe)

final_dataframe.to_csv(final_combined, index=False)
