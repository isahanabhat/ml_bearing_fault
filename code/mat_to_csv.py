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


sourceFiles = [[h_1797_12k_normal, 0], [f_1797_12k_de_ir007, 1], [f_1797_12k_de_b007, 2],
               [f_1797_12k_de_or007_3, 3] , [f_1797_12k_de_or007_6, 4] , [f_1797_12k_de_or007_12, 5]]
destFiles = [h_1797_12k_normal_csv, f_1797_12k_de_ir007_csv, f_1797_12k_de_b007_csv,
             f_1797_12k_de_or007_3_csv, f_1797_12k_de_or007_6_csv, f_1797_12k_de_or007_12_csv]

faulty_combined = r"../csv_data/faulty_combined_1797_12k.csv"

final_combined = r"../csv_data/combined_1797_12k.csv"

# de -> drive end accelerometer data
# fe -> fan end accelerometer data

datalist = []


def matTocsv(source, destination, status):
    dataset = scipy.io.loadmat(source)
    # print(type(dataset))
    header = list(dataset.keys())
    # print(header)

    col = []
    for i in header[::-1]:
        if i == '__globals__':
            break
        col.append(i)
    # print(col)

    df = pd.DataFrame()
    for i in col[1:]:
        # print(dataset[i])
        col_name = i.split("_", 1)[1]
        df[col_name] = dataset[i].tolist()
    df['Y'] = status

    df = df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    df["DE_time (MA)"] = df["DE_time"].rolling(window=5).mean()
    df["FE_time (MA)"] = df["FE_time"].rolling(window=5).mean()
    df["DE_time (SD)"] = df["DE_time"].rolling(window=5).std()
    df["FE_time (SD)"] = df["FE_time"].rolling(window=5).std()

    df = df.fillna(0)
    """de = dataset[header[-3]]
    fe = dataset[header[-2]]
    rpm = dataset[header[-1]]

    de_time = []
    fe_time = []
    rpm_val = []
    for i in range(0, len(de)):
        de_time.append(de[i][0])
        fe_time.append(fe[i][0])

    for i in rpm:
        for j in i:
            rpm_val.append(j)

    dataFrame = pd.DataFrame(de_time, columns=['X097_DE_time'])
    dataFrame['X097_FE_time'] = fe_time
    dataFrame['Y'] = status"""

    datalist.append(df)
    df.to_csv(destination, index=False)
    print("Succesful!")

matTocsv(sourceFiles[0][0], destFiles[0], sourceFiles[0][1])

for i in range(1,6):
    matTocsv(sourceFiles[i][0], destFiles[i], sourceFiles[i][1])

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


"""# 12k Fan End Bearing Fault Data
dataset = scipy.io.loadmat(file_12kfanEndFault_innerRace[0])
header = list(dataset.keys())
print(header)

de = dataset[header[-3]]
fe = dataset[header[-2]]
rpm = dataset[header[-1]]

de_time = []
fe_time = []
rpm_val = []

for i in range(0, len(de)):
    de_time.append(de[i][0])
    fe_time.append(fe[i][0])

for i in rpm:
    for j in i:
        rpm_val.append(j)

df3 = pd.DataFrame(de_time, columns=['X097_DE_time'])
df3['X097_FE_time'] = fe_time

with pd.ExcelWriter("..\data\online\sample_data.xlsx") as writer:
    df1.to_excel(writer, sheet_name="healthy bearing")
    df2.to_excel(writer, sheet_name="48k drive end fault")
    # df3.to_excel(writer, sheet_name="12k fan end fault")
"""