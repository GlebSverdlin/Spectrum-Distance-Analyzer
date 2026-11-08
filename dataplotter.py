from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import json
from secret import *
from dataloader import *
import astropy
from astropy.io import fits
from astropy.table import Table
import matplotlib.patches as ptc
from matplotlib import ticker


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

def get_teff_logg(names, path_1, path_2):
    teffs = []
    loggs = []
    fits_list = []
    for name in names:
        fits_list.append(name.replace('.csv', '.fits'))

    for dir in os.scandir(path_1):
        for file in os.scandir(os.path.join(path_1, dir)):
            if file.name in fits_list:
                data = fits.open(os.path.join(path_1, dir, file.name)) 
                teffs.append(str(data[4].data['FPARAM']).split()[1])
                loggs.append(str(data[4].data['FPARAM']).split()[2])
              
    for dir in os.scandir(path_2):
        for file in os.scandir(os.path.join(path_2, dir)):
            if file.name in fits_list:
                data = fits.open(os.path.join(path_1, dir, file.name))
                teffs.append(float(str(data[4].data['FPARAM']).split()[1]))
                loggs.append(float(str(data[4].data['FPARAM']).split()[2]))
    return teffs, loggs
 

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


    teff_tp_1, logg_tp_1 = get_teff_logg(true_positives[1], k2_pos, kep_pos)
    teff_tn_1, logg_tn_1 = get_teff_logg(true_negatives[1], k2_neg, kep_neg)
    teff_fn_1, logg_fn_1 = get_teff_logg(false_negatives[1], k2_pos, kep_pos)
    teff_fp_1, logg_fp_1 = get_teff_logg(false_positives[1], k2_neg, kep_neg)
    teff_tp_2, logg_tp_2 = get_teff_logg(true_positives[1], k2_neg, kep_neg)
    teff_tn_2, logg_tn_2 = get_teff_logg(true_negatives[1], k2_pos, kep_pos)
    teff_fn_2, logg_fn_2 = get_teff_logg(false_negatives[1], k2_neg, kep_neg)
    teff_fp_2, logg_fp_2 = get_teff_logg(false_positives[1], k2_pos, kep_pos)
    
    teff_tp = teff_tp_1+teff_tp_2
    teff_tn = teff_tn_1+teff_tn_2
    teff_fp = teff_fp_1+teff_fp_2
    teff_fn = teff_fn_1+teff_fn_2



    logg_tp = logg_tp_1+logg_tp_2
    logg_tn = logg_tn_1+logg_tn_2
    logg_fp = logg_fp_1+logg_fp_2
    logg_fn = logg_fn_1+logg_fn_2

    print(len(teff_fn))

    plt.style.use('bmh')

    fig, ax = plt.subplots(1,1)

    delta = '\\Delta_{err}'
    
    ax.scatter(teff_tp, logg_tp, color='r', marker = 'x', label = 'True positive samples')
    ax.scatter(teff_fp, logg_fp, color='g', marker = 'o', label = 'False positive samples')
    ax.scatter(teff_tn, logg_tn, color='b', marker = 'x', label = 'True negative samples')
    ax.scatter(teff_fn, logg_fn, color='k', marker = 'o', label = 'False negative samples')
    ax.set_title('$T_{eff}$ to $log_{g}$ relation between 4 sample groups')

    # ax[1].scatter(teff_tp, logg_tp, color='r', marker = 'o')
    # ax[1].scatter(teff_fp, logg_fp, color='g', marker = '^')
    # ax[1].scatter(teff_tn, logg_tn, color='b', marker = '^')
    # ax[1].scatter(teff_fn, logg_fn, color='k', marker = 'o')
    # ax[1].set_title('$T_{eff}$ to $log_{g}$ with hypothetical 100% accuracy')


    # patch_m = ptc.Rectangle((2400, 0), 1300, 6, facecolor = 'darkorange', alpha=0.2)
    # patch_k = ptc.Rectangle((3700, 0), 1500, 6, facecolor = 'yellow', alpha=0.2)
    # patch_g = ptc.Rectangle((5200, 0), 800, 6, facecolor = 'lemonchiffon', alpha=0.2)
    # patch_f = ptc.Rectangle((6000, 0), 1500, 6, facecolor = 'lightsteelblue', alpha=0.2)
    # patch_a = ptc.Rectangle((7500, 0), 2500, 6, facecolor = 'deepskyblue', alpha=0.2)
    # patch_b = ptc.Rectangle((10000, 0), 20000, 6, facecolor = 'blue', alpha=0.2)
    #
    # ax.add_patch(patch_b)
    # ax.add_patch(patch_a)
    # ax.add_patch(patch_f)
    # ax.add_patch(patch_g)
    # ax.add_patch(patch_k)
    # ax.add_patch(patch_m)
    # ax.text(2450, 5.5, 'M', size='x-large', weight='bold')
    # ax.text(3750, 5.5, 'K', size='x-large', weight='bold')
    # ax.text(5250, 5.5, 'G', size='x-large', weight='bold')
    # ax.text(6050, 5.5, 'F', size='x-large', weight='bold')
    # ax.text(7550, 5.5, 'A', size='x-large', weight='bold' )
    # ax.text(10050, 5.5, 'B', size='x-large', weight='bold')





    ax.set_xlim(2000, 16000)
    ax.set_ylim(0, 6)
    ax.set_xlabel('T$_{eff}$')
    ax.set_ylabel('log$_{g}$')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(250))
    # ax[1].set_xlim(2000, 16000)
    # ax[1].set_ylim(0, 6)
    # ax[1].set_xlabel('T$_{eff}$')
    # ax[1].set_ylabel('log$_{g}$')
    # ax.set_title('$T_{eff}$ to $log_{g}$ relation between 4 sample groups.')
    ax.legend()
    
    

    # ax[0,0].scatter(np.linspace(0, len(false_negatives[0]), num = len(false_negatives[0])), false_negatives[0], color = 'r')
    # ax[1,0].scatter(np.linspace(0, len(true_negatives[0]), num = len(true_negatives[0])), true_negatives[0])
    # ax[0,0].set_title(f'False Negatives, {len(false_negatives[0])} samples.')
    # ax[1,0].set_title(f'True Negatives ($%s$ < 0.25), {len(true_negatives[0])} samples.'%delta)
    #
    # ax[0,1].scatter(np.linspace(0, len(false_positives[0]), num = len(false_positives[0])), false_positives[0], color = 'r')
    # ax[1,1].scatter(np.linspace(0, len(true_positives[0]), num = len(true_positives[0])), true_positives[0])
    # ax[0,1].set_title(f'False Positives, {len(false_positives[0])} samples.')
    # ax[1,1].set_title(f'True Positives ($%s$ < 0.25), {len(true_positives[0])} samples.'%delta)
    #
    # for i in false_negatives[2]:
    #     ax[0,0].plot(spectra[i][0], alpha = 1/len(false_negatives), color = 'b')
    #
    # for i in true_negatives[2]:
    #     ax[1,0].plot(spectra[i][0], alpha = 1/len(true_negatives), color = 'b')
    #
    # for i in false_positives[2]:
    #     ax[0,1].plot(spectra[i][0], alpha = 1/len(false_positives), color = 'b')
    #
    # for i in true_positives[2]:
    #     ax[1,1].plot(spectra[i][0], alpha = 1/len(true_positives), color = 'b')
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
