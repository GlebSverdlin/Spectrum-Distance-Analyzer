import os
import csv
import re
from itertools import zip_longest
from secrets import *

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

def unpack_ap_first(file_params):
    unpack_list = []
    appended = []
    for file in file_params:
        name = file[1]
        if re.search('apStar|asStar', name) != None:
            unpack_list.append(file[0])

    return unpack_list

def write_as_table(wave, flux, planet, name):
    colnames = [['wave', 'flux', 'planet' ]]
    rows = zip_longest(wave,flux,planet, fillvalue = '')
    path = str(dataset+name)
    
    with open(path, mode='w', newline ='') as file:
        writer=csv.writer(file)
        writer.writerows(colnames)


    with open(path, mode='a', newline ='') as file:
        writer=csv.writer(file)
        writer.writerows(rows)


