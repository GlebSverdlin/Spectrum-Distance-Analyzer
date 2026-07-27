import pandas as pd
import csv

stellarhosts = 'STELLARHOSTS_2026.07.22_02.07.13.csv'
stellar_output = 'Stellar.csv'

host_data = pd.read_csv(stellarhosts, usecols = ['sy_name', 'hostname', 'sy_snum' ,'sy_pnum'])

with open(stellar_output, 'w', newline='') as datafile:
    fieldnames = ['star'] 
    writer = csv.DictWriter(datafile, fieldnames=fieldnames)
    writer.writeheader()

for i in host_data['sy_pnum']:
    if i == 1:
        print(host_data[])


with open(stellar_output, 'a', newline='') as datafile:
            fieldnames = ['star'] 
            

