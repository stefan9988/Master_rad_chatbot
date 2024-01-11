import pandas as pd

data = pd.read_csv('data/insurance_qna_dataset.csv', delimiter='\t', index_col=0)
u = data.Question.unique()

row = {}

for j in range(u.shape[0]):
    row[j] = []

for i in range(data.shape[0]):
    value = data.iloc[i, 0]
    if value in u:
        row[u.tolist().index(value)].append(i)

df = pd.DataFrame()
df['Question'] = u
df['Answer'] = u
for i in range(len(row)):
    row_values = row[i]
    string = ''

    for j in row_values:
        string += data.iloc[j, 1]

    df.iloc[i, 1] = string

print(df.shape)

file_path = 'data/data_unique.csv'
# df.to_csv(file_path, index=False)
