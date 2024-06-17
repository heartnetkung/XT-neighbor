from os import path, listdir
import pandas as pd

INPUT_FOLDER = path.abspath(
    path.join(__file__, '../../../msc_proj/data/emerson'))

count = 0
count18 = 0

for i, file in enumerate(listdir(INPUT_FOLDER)):
    absfile = path.abspath(path.join(INPUT_FOLDER, file))
    df = pd.read_csv(absfile, sep='\t', usecols=['amino_acid'])
    df.dropna(inplace=True)
    df = df[df['amino_acid'].str.match(
        r'^[ACDEFGHIKLMNPQRSTVWY]+$')]
    count += len(df)
    count18 += len(df[df['amino_acid'].str.len()>18])
    print(i, count,count18)

print("result",count,count18)
