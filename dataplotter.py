from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import json
from secret import *
from dataloader import *

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

def plot_eval(path, spectra):    
    with open(path, 'r') as file:
        data = json.load(file)

    answers = data['answers']
    lables = data['corrects']
    errors = data['errors'] 
    
    true_positives = [[],[],[]]
    true_negatives = [[],[],[]]
    false_positives = [[],[],[]]
    false_negatives = [[],[],[]]

    for i in answers:
        if lables[answers.index(i)] == 1:
            if errors[answers.index(i)] <= 0.25:
                true_positives[0].append(i)
                true_positives[1].append(spectra.getname(answers.index(i)))
                true_positives[2].append(answers.index(i))

            else:
                false_positives[0].append(i)
                false_positives[1].append(spectra.getname(answers.index(i)))
                false_positives[2].append(answers.index(i))

        else:
            if errors[answers.index(i)] <= 0.25:
                true_negatives[0].append(i)
                true_negatives[1].append(spectra.getname(answers.index(i)))
                true_negatives[2].append(answers.index(i))

            else:
                false_negatives[0].append(i)
                false_negatives[1].append(spectra.getname(answers.index(i)))
                false_negatives[2].append(answers.index(i))

    
    print(true_negatives[1])

    plt.style.use('bmh')

    fig, ax = plt.subplots(2,2)

    delta = '\\Delta_{err}'
    
    # ax[0,0].scatter(np.linspace(0, len(false_negatives[0]), num = len(false_negatives[0])), false_negatives[0], color = 'r')
    # ax[1,0].scatter(np.linspace(0, len(true_negatives[0]), num = len(true_negatives[0])), true_negatives[0])
    ax[0,0].set_title(f'False Negatives, {len(false_negatives[0])} samples.')
    ax[1,0].set_title(f'True Negatives ($%s$ < 0.25), {len(true_negatives[0])} samples.'%delta)
    #
    # ax[0,1].scatter(np.linspace(0, len(false_positives[0]), num = len(false_positives[0])), false_positives[0], color = 'r')
    # ax[1,1].scatter(np.linspace(0, len(true_positives[0]), num = len(true_positives[0])), true_positives[0])
    ax[0,1].set_title(f'False Positives, {len(false_positives[0])} samples.')
    ax[1,1].set_title(f'True Positives ($%s$ < 0.25), {len(true_positives[0])} samples.'%delta)

    for i in false_negatives[2]:
        ax[0,0].plot(spectra[i][0], alpha = 1/len(false_negatives), color = 'b')

    for i in true_negatives[2]:
        ax[1,0].plot(spectra[i][0], alpha = 1/len(true_negatives), color = 'b')
  
    for i in false_positives[2]:
        ax[0,1].plot(spectra[i][0], alpha = 1/len(false_positives), color = 'b')

    for i in true_positives[2]:
        ax[1,1].plot(spectra[i][0], alpha = 1/len(true_positives), color = 'b')
    #
    # ax[0,0].set_ylim(0, 3)
    # ax[1,0].set_ylim(0, 3)
    # ax[0,1].set_ylim(0, 3)
    # ax[1,1].set_ylim(0, 3)

    #
    # plt.plot(np.linspace(0, len(answers), num = len(answers)), corrects, '_')
    # plt.scatter(np.linspace(0, len(answers), num = len(answers)), answers, c = errors, cmap = 'plasma', marker='x')
    # plt.colorbar()
    # plt.legend(['Average: '+str(average)])
    #
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
option = "p:"
long_option = ["path"]

args, vals = getopt.getopt(args,option, long_option)

try:
    for arg, val in args:
        if arg in ("-p", "--path"):
            json_path = sys.argv[2]

except: print("ERR")

eval_data = SpectralDataset('aspcap', 'eval')
plot_eval(json_path, eval_data)
