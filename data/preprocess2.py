from os import path, listdir
import pandas as pd
import time

INPUT_FOLDER = path.abspath(
    path.join(__file__, '../../../msc_proj/data/emerson'))
output_count = 1
count = 0
cum_freq = 0
data = {}
start = time.time()

for i, file in enumerate(listdir(INPUT_FOLDER)):
    absfile = path.abspath(path.join(INPUT_FOLDER, file))
    df = pd.read_csv(absfile, sep='\t', usecols=['amino_acid','templates'])
    df.dropna(inplace=True)
    df = df[df['amino_acid'].str.match(
        r'^[ACDEFGHIKLMNPQRSTVWY]{1,18}$')]
    aa_list = df['amino_acid'].tolist()
    template_list = df['templates'].tolist()

    for j in range(len(aa_list)):
        amino_acid, template = aa_list[j], template_list[j]
        if data.get(amino_acid) is None:
            data[amino_acid] = template
        else:
            data[amino_acid] += template
        count+=1
        cum_freq+=template
    if i%10==0:
        print(i)

print("count:",f'{count:,}')
print("cum_freq:",f'{cum_freq:,}')
print(len(data))

aa_buffer = []
template_buffer =[]

for amino_acid,template in data.items():
    aa_buffer.append(amino_acid)
    template_buffer.append(template)
    if len(aa_buffer)==10_000_000:
        thefile = path.abspath(path.join(__file__, f'../emerson{output_count}.txt'))
        pd.DataFrame({'cdr3':aa_buffer,'count':template_buffer}).to_csv(thefile,index=False)
        aa_buffer=[]
        template_buffer=[]
        output_count+=1

if len(aa_buffer)>0:
    thefile = path.abspath(path.join(__file__, f'../emerson{output_count}.txt'))
    pd.DataFrame({'cdr3':aa_buffer,'count':template_buffer}).to_csv(thefile,index=False)

print("elapsed time:",f'{time.time()-start:,}')