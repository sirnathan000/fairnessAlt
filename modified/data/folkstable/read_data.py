import pandas as pd


file_path = 'folkstable.csv'


df = pd.read_csv(file_path, sep=';')

print(df.head())

# Display the DataFrame
data_sorted = sorted(df['COW'].unique().tolist())

for data in data_sorted:
    print(f"{data};")