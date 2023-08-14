import pandas as pd
file1 = r"C:\Users\User\Downloads\train-00000-of-00034-9aa97ead99fa5193.parquet"
df = pd.read_parquet(file1)
df.head()
csv_file1 = r"C:\Users\User\Downloads\file1.csv"
df.to_csv(csv_file1,index = False)