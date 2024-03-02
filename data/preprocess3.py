from os import path, listdir
import pandas as pd
import time

INPUT_FOLDER = path.abspath(
    path.join(__file__, '../../../msc_proj/data/emerson'))
output_count = 1
count = 0
cum_freq = 0
start = time.time()
buffer = {'cdr3':[],'count':[]}
info = {'file':[], 'count':[]}

for i, file in enumerate(listdir(INPUT_FOLDER)):
    absfile = path.abspath(path.join(INPUT_FOLDER, file))
    df = pd.read_csv(absfile, sep='\t', usecols=['amino_acid','templates'])
    df.dropna(inplace=True)

    print(i,file,df.shape)
    df = df[df['amino_acid'].str.match(
        r'^[ACDEFGHIKLMNPQRSTVWY]{1,18}$')]
    aa_list = df['amino_acid'].tolist()
    template_list = df['templates'].tolist()
    data = {}

    for j in range(len(aa_list)):
        amino_acid, template = aa_list[j], template_list[j]
        if data.get(amino_acid) is None:
            data[amino_acid] = template
        else:
            data[amino_acid] += template
        count+=1
        cum_freq+=template

    for cdr3, freq in data.items():
        buffer['cdr3'].append(cdr3)
        buffer['count'].append(freq)

    info['file'].append(file)
    info['count'].append(len(data))

    if i%50 == 49:
        thefile = path.abspath(path.join(__file__, f'../emerson_rep{output_count}.txt'))
        pd.DataFrame(buffer).to_csv(thefile,index=False)
        buffer = {'cdr3':[],'count':[]}
        output_count+=1

if len(buffer['cdr3'])>0:
    thefile = path.abspath(path.join(__file__, f'../emerson_rep{output_count}.txt'))
    pd.DataFrame(buffer).to_csv(thefile,index=False)

thefile = path.abspath(path.join(__file__, f'../info.csv'))
pd.DataFrame(info).to_csv(thefile,index=False)
print("count:",f'{count:,}')
print("cum_freq:",f'{cum_freq:,}')
print(len(data))
print("elapsed time:",f'{time.time()-start:,}')