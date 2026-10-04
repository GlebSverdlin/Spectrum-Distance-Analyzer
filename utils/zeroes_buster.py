from secret import *
import pandas as pd
import numpy as np
import os
import csv
from itertools import zip_longest

od = dataset_aspcap
id = dataset_aspcap_nz

tables = os.scandir(od)
tables_arr = []
labels_arr = []
waves_arr = []
#loading all spectra:
for table in tables:
    tables_arr.append(pd.read_csv(os.path.join(od, table.name), usecols=["flux"], index_col=False)['flux'].tolist())
    labels_arr.append(pd.read_csv(os.path.join(od, table.name), usecols=["planet"], index_col=False)['planet'].tolist())
    waves_arr.append(pd.read_csv(os.path.join(od, table.name), usecols=["wave"], index_col=False)["wave"].tolist())

nonzero_idx = np.nonzero(tables_arr[0])

if len(np.nonzero(tables_arr[1]))!=len(nonzero_idx):
    print(f'ZEROING NOT IDENTIC FOR TABLE {tables_arr.index(table)}')

print('Zeroing checked: OK')

tables_clear = []

for table in tables_arr:
    table_clear = []
    for index in nonzero_idx[0]:
        table_clear.append(table[index])
    print(len(table_clear))
    if len(table_clear) == 0: print(f"ERROR on table {tables_arr.index(table)}"); break
    tables_clear.append(table_clear)
print(len(tables_clear))

print('Zeroes removed: OK')

vals_global = []

for table in tables_clear:
    for val in table:
        vals_global.append(val)

global_mean = np.mean(vals_global)
global_std = np.std(vals_global)

print(f"Globals: {global_mean}, {global_std}")

for table in tables_clear:
    for value in table:
        value = (value-global_mean)/global_std

for iter, file in enumerate(tables):
    path = os.path.join(id, file.name)
    
    flux = tables_clear[iter]
    planet = labels_arr[iter]
    wave = waves_arr[iter]
    
    colnames = [['wave', 'flux', 'planet' ]]
    rows = zip_longest(wave,flux,planet, fillvalue = '')

    with open(path, mode='w', newline ='') as table:
        writer=csv.writer(table)
        writer.writerows(colnames)


    with open(path, mode='a', newline ='') as table:
        writer=csv.writer(table)
        writer.writerows(rows)
print(f"Table {iter} written: OK")

print('Dataset written: OK')
        
    # with open(os.path.join(od,table.name)) as spectrum:
    #     data = csv.reader(spectrum, delimiter=',')
    #     for row in data:
    #         print(row[0])
    #         if row[0] == 0:
    #             print(f'ZERO DETECTED ON LINE {data.index(row)}')
    #     break
