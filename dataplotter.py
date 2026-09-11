from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(losses):
    tr = []
    for i in losses:
        if losses.index(i)%2!=0:
            tr.append(i)
    max = np.max(tr)
    min = np.min(tr)
    avrg = np.average(tr)
    max_idx = tr.index(max)
    min_idx = tr.index(min)
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,7))

    tr_filter = savgol_filter(tr, window_length = int(len(tr)/10), polyorder = 3)
    #, ['max', 'max_idx', 'min', 'min_idx', 'average'])
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr_filter, color = 'r')
    plt.plot(1,1,color='w')
    plt.legend([str("Max: " + str(max)+'; Idx: '+str(max_idx)), str('Min: ' + str(min)+'; Idx: '+str(min_idx)),str('Avrg: '+ str(avrg))])
    plt.show()
