from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import json

def plot_losses(losses, save):
    tr = []
    for i in losses:
        if losses.index(i)%2!=0:
            tr.append(i)
    max = np.max(tr)
    min = np.min(tr)
    avrg = np.average(tr)
    max_idx = tr.index(max)
    min_idx = tr.index(min)
    median = np.median(tr)
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,7))

    tr_filter = savgol_filter(tr, window_length = int(len(tr)/10), polyorder = 3)
    #, ['max', 'max_idx', 'min', 'min_idx', 'average'])
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr_filter, color = 'r')
    plt.plot(1,1,color='w')
    plt.plot(1,1,color='w')
    plt.legend([str("Max: " + str(max)+'; Idx: '+str(max_idx)), str('Min: ' + str(min)+'; Idx: '+str(min_idx)),str('Avrg: '+ str(avrg)), str('Mdn: '+str(median))])
    plt.show()

def plot_eval(data):
    answers = []
    corrects = []
    
    for i in data:
        if data.index(i)%2!=0:
            answers.append(i)
            corrects.append(data[data.index(i)-1])
    
    plt.style.use('bmh')
    plt.plot(np.linspace(0, len(answers), num = len(answers)), corrects, '_')
    plt.scatter(np.linspace(0, len(answers), num = len(answers)), answers, c = errors, cmap = 'plasma', marker='x')
    plt.colorbar()
    plt.legend(['Average: '+str(average)])

    plt.show()

def plot_from_json(path):
    with open(path, 'r') as file:
        data = json.load(file)

    losses = data['losses']

    max = np.max(losses)
    min = np.min(losses)
    avrg = np.average(losses)
    max_idx = losses.index(max)
    min_idx = losses.index(min)
    median = np.median(losses)
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,7))

    losses_filter = savgol_filter(losses, window_length = int(len(losses)/10), polyorder = 3)
    #, ['max', 'max_idx', 'min', 'min_idx', 'average'])
    plt.plot(np.linspace(0, len(losses), num = len(losses)), losses)
    plt.plot(np.linspace(0, len(losses), num = len(losses)), losses_filter, color = 'r')
    plt.plot(1,1,color='w')
    plt.plot(1,1,color='w')
    plt.legend([str("Max: " + str(max)+'; Idx: '+str(max_idx)), str('Min: ' + str(min)+'; Idx: '+str(min_idx)),str('Avrg: '+ str(avrg)), str('Mdn: '+str(median))])
    plt.show()


args = sys.argv[1:]
option = "m:"
long_option = ["model"]

args, vals = getopt.getopt(args,option, long_option)

try:
    for arg, val in args:
        if arg in ("-m", "--model"):
            json_path = sys.argv[2]

except: print("ERR")
plot_from_json(json_path)
