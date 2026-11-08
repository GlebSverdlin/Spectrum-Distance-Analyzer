import torch
from dataloader import *
import os
import datetime
from secret import *
from torch.utils.data import DataLoader
import sys
import getopt
from network import NeuralNetwork
import time
import matplotlib.pyplot as plt
import numpy as np
import shap
import random
from matplotlib import ticker
import csv
from itertools import zip_longest

args = sys.argv[1:]
option = "m:"
long_option = ["model"]

args, vals = getopt.getopt(args,option, long_option)

try:
    for arg, val in args:
        if arg in ("-m", "--model"):
            model_name = sys.argv[2]
            print(model_name)
            path = PATH+model_name

except: print(str(getopt.error))

nist_lines = os.path.join(test_env_dir, "nist_output_lines.csv")
lines_air_obs_raw = pd.read_csv(nist_lines, usecols=['obs_wl_air(A)'], index_col=False).to_numpy(dtype=np.double, na_value=0.0)
lines_air_obs=np.reshape(lines_air_obs_raw, len(lines_air_obs_raw)).tolist()
elements = pd.read_csv(nist_lines, usecols=['element'], index_col=False).values.tolist()
ions = pd.read_csv(nist_lines, usecols=['sp_num'], index_col=False).values.tolist()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

wave_data = SpectralDataset('aspcap', 'eval')
eval_data = SpectralDataset('research_tp', 'res')
eval_data_2 = SpectralDataset('research_tn', 'res')
eval_data.concatenate(eval_data_2)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(path, weights_only = True, map_location=torch.device('cpu')))

date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+date)

log_path = os.path.join(logs_shap, str(data_name+'-test_date-'+date))
os.mkdir(log_path) 

eval_loader = DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
model.eval()

for iter, batch in enumerate(eval_loader):
    #
    # random.shuffle(batch)
    background_data = batch[0][30:len(eval_data)-30]
    test_data_neg = batch[0][:30]
    test_data_pos = batch[0][len(eval_data)-30:len(eval_data)]


    print(f"Shapes: tp:{np.shape(test_data_pos)}, tn:{np.shape(test_data_neg)}, bg:{np.shape(background_data)}")

    explainer = shap.DeepExplainer(model, background_data)

    shap_vals_pos = explainer.shap_values(test_data_pos)
    shap_vals_neg = explainer.shap_values(test_data_neg)

    shap_vals = list(shap_vals_pos)+list(shap_vals_neg)
    
    average_values = []
    average_values_pos = []
    average_values_neg = []

    for value in range(len(shap_vals[0])):
        line_unav = []
        for line in shap_vals:
            line_unav.append(line[value])
        average_values.append(np.average(line_unav).tolist())

        line_unav_pos = []
        line_unav_neg = []
        for line in shap_vals_pos:
            line_unav_pos.append(line[value])
        average_values_pos.append(np.average(line_unav_pos).tolist())
        for line in shap_vals_neg:
            line_unav_neg.append(line[value])
        average_values_neg.append(np.average(line_unav_neg).tolist())


    zero_seq = []
    for line in average_values:
        if line == 0:
            zero_seq.append(list(average_values).index(line))

    print(np.shape(zero_seq))

    std_dev = np.std(np.delete(average_values, zero_seq))
    
    p = []
    n = []

    for i in shap_vals_pos[0]:
        p.append(0+i)
    for i in shap_vals_neg[0]:
        n.append(0+i)


    c = []
    for i in average_values:
        c.append(0+i)

    peaks_positive = []
    peaks_negative = []
    for i in average_values:
        if i >= 3*std_dev:
            peaks_positive.append(average_values.index(i))
        if i <= -3*std_dev:
            peaks_negative.append(average_values.index(i))

    print(f"positive idx: {peaks_positive}")
    print(f"negative idx: {peaks_negative}") 

    wave = wave_data.get_wave()



    positive_lines = []
    negative_lines = []

    for i in peaks_positive:
        positive_lines.append(wave[i][0])
    for i in peaks_negative:
        negative_lines.append(wave[i][0])

    # print(f"positive waves: {positive_lines}")
    # print(f"negative waves: {negative_lines}") 

    fig, ax = plt.subplots(4,1)
    plt.style.use('bmh')

    average_pos_neg = []

    for i in range(len(average_values_pos)):
        average_pos_neg.append((average_values_pos[i]+average_values_neg[i])/2)
    print(len(average_pos_neg))
    image = ax[0].scatter(np.linspace(0, len(average_values), num = len(average_values)), average_pos_neg, c = average_pos_neg, cmap = 'coolwarm', marker = 'x')

    ax[0].axhline(y=3*std_dev, color = 'lime')
    ax[0].axhline(y=-3*std_dev, color = 'lime')

    # ax.plot(np.linspace(0, len(average_values), num = len(average_values)), np.full(shape = len(average_values), fill_value= 3*std_dev), color = 'lime')
    # ax.plot(np.linspace(0, len(average_values), num = len(average_values)), np.full(shape = len(average_values), fill_value= -3*std_dev), color = 'lime') 
    # fig.colorbar(image) 
    
    for line in positive_lines:
        ax[3].axvline(x=line, color='r', alpha = 150*c[peaks_positive[positive_lines.index(line)]])
    for line in negative_lines:
        ax[3].axvline(x=line, color='b', alpha = -150*c[peaks_negative[negative_lines.index(line)]])
    



    ax[1].scatter((np.linspace(0, len(average_values), num = len(average_values))), average_values_pos, c=average_values_pos, cmap='coolwarm', marker='x')
    ax[2].scatter((np.linspace(0, len(average_values), num = len(average_values))), average_values_neg, c=average_values_neg, cmap='coolwarm', marker='x')
    

    pos_els=[]
    neg_els=[]
    j = 0
    k = 0
    rows = []
    print("Positive elements:") 
    for line in positive_lines:
        row = [line, c[peaks_positive[positive_lines.index(line)]], '+']
        rows.append(row)
        for wave in lines_air_obs:
            if np.isclose(line, wave):
                ax[3].axvline(x=line, color='w')
                ax[3].axvline(x=line, color='r', ls="--", alpha = 150*c[peaks_positive[positive_lines.index(line)]])
                element = elements[lines_air_obs.index(wave)]+ions[lines_air_obs.index(wave)]
                importance = c[peaks_positive[positive_lines.index(line)]]
                print(element, importance, wave)
                pos_els.append([element, importance, wave])
                
                j+=1



    print("Negative elements:")
    for line in negative_lines:
        row = [line, c[peaks_negative[negative_lines.index(line)]], '-']
        rows.append(row)
        for wave in lines_air_obs:
            if np.isclose(line, wave):
                ax[3].axvline(x=line, color='w')
                ax[3].axvline(x=line, color='b', ls="--", alpha = -150*c[peaks_negative[negative_lines.index(line)]])
                element = elements[lines_air_obs.index(wave)]+ions[lines_air_obs.index(wave)]
                importance = -c[peaks_negative[negative_lines.index(line)]]
                print(element, importance, wave)
                neg_els.append([element, importance, wave])
                k+=1


    colnames = [['lambda'],['importance'],['pos/neg']]

    with open(os.path.join(test_env_dir, 'wavelenghts.csv'), mode='w', newline='') as file:
        writer=csv.writer(file)
        writer.writerows(colnames)
     
    with open(os.path.join(test_env_dir, 'wavelenghts.csv'), mode='a', newline ='') as file:
        writer=csv.writer(file)
        writer.writerows(rows)


    print(f'Found {j+k} elements out of {len(positive_lines)+len(negative_lines)} lines.')
    print("Unique pos. elements:")
    print(pos_els)
    print("Unique neg. elements:")
    print(neg_els)

 
    ax[3].set_yticks([])

    # plt.savefig(f"{log_path}/fig.pdf")
    plt.show()
    break

    
# with open(f"{log_path}/log.txt", 'x') as file:
#     file.write(f'Model: {model_name}\n')
#     file.write(f'Average predicted effect of each feature\n')
#     file.write(str(average_values))

