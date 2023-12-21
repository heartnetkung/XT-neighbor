from os import path, listdir
import pandas as pd

INPUT_FOLDER = path.abspath(
    path.join(__file__, '../../../msc_proj/data/emerson'))
outputCount = 1

output = open(path.abspath(path.join(__file__, f'../emerson{outputCount}.txt')), 'w')
count = 0

for i, file in enumerate(listdir(INPUT_FOLDER)):
    absfile = path.abspath(path.join(INPUT_FOLDER, file))
    df = pd.read_csv(absfile, sep='\t', usecols=['amino_acid'])
    df.dropna(inplace=True)
    df = df[df['amino_acid'].str.match(
        r'^[ACDEFGHIKLMNPQRSTVWY]{1,18}$')]
    lines = df['amino_acid'].tolist()
    count += len(lines)
    print(f"{i} {count:,}")
    output.writelines(line+'\n' for line in lines)

    if i%50 == 49:
    	output.close()
    	outputCount += 1
    	output = open(path.abspath(path.join(__file__, f'../emerson{outputCount}.txt')), 'w')

output.close()
