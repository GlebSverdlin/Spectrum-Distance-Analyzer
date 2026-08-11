import os
import csv
import re
from itertools import zip_longest
from secret import *
from apogee_multifunc_unpacker import *

def scan_dir(dirpath):
    dirs = []
    files = []
    for obj in os.scandir(dirpath):
        if obj.is_file(): files.append([obj.path, obj.name])    
        else: dirs.append([obj.path, obj.name])
    
    return files, dirs

def filetype(file_params):
    path = file_params[0]
    name = file_params[1]
    if re.search('aspcapStar', name) != None: filetype = 'aspcap'
    else: filetype = 'ap'
    return [path, name, filetype]

def write_as_table(wave, flux, planet, name, dir):
    match dir:
        case 'ap':targ_dir = dataset_ap 
        case 'aspcap':targ_dir = dataset_aspcap

    colnames = [['wave', 'flux', 'planet' ]]
    rows = zip_longest(wave,flux,planet, fillvalue = '')
    path = str(targ_dir+name.replace('.fits', '.csv'))
    
    with open(path, mode='w', newline ='') as file:
        writer=csv.writer(file)
        writer.writerows(colnames)


    with open(path, mode='a', newline ='') as file:
        writer=csv.writer(file)
        writer.writerows(rows)


#----------------------------------

def exec_table_pipeline(path):

    files, dirs = scan_dir(path)
    
    if re.search('pos', path): planet = [1]
    else: planet = [0]
    print(len(dirs))

    for dir in dirs:
        name = dir[1]
        files, subdirs = scan_dir(dir[0])

        for file in files:
            params = filetype(file)
            flux_raw, wave = load_fits(params[0], params[2])
            if params[2] == 'ap': flux_norm = normalize_spectrum(wave, flux_raw)
            flux_norm = rescale_spectrum(flux_norm)
            write_as_table(wave, flux_norm, planet, params[1], str(params[2]))




exec_table_pipeline(k2_neg)            
exec_table_pipeline(k2_pos)
exec_table_pipeline(kep_pos)
exec_table_pipeline(kep_neg)
               
