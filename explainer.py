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

    for value in range(len(shap_vals[0])):
        line_unav = []
        for line in shap_vals:
            line_unav.append(line[value])
        average_values.append(np.average(line_unav).tolist())

    zero_seq = []
    for line in average_values:
        if line == 0:
            zero_seq.append(list(average_values).index(line))

    print(np.shape(zero_seq))

    std_dev = np.std(np.delete(average_values, zero_seq))

    c = []
    for i in average_values:
        c.append(0+i)

    peaks = []
    for i in average_values:
        if i >= 3*std_dev:
            peaks.append(average_values.index(i))

    print(peaks)
    print(len(peaks))

    wave = wave_data.get_wave()

    zero_idx = np.linspace(0,245,num=246).tolist()+np.linspace(3274,3584,num=3584-3274+1).tolist()+np.linspace(6080,6343, num = 6343-6080+1).tolist()+np.linspace(8335,8575,num=8575-8335).tolist()
    print(len(wave))

    fig, ax = plt.subplots(1,1)
    plt.style.use('bmh')

    image = ax.scatter(np.linspace(0, len(average_values), num = len(average_values)), average_values, c = c, cmap = 'plasma', marker = 'x')

    ax.axhline(y=3*std_dev, color = 'lime')
    ax.axhline(y=-3*std_dev, color = 'lime')

    # ax.plot(np.linspace(0, len(average_values), num = len(average_values)), np.full(shape = len(average_values), fill_value= 3*std_dev), color = 'lime')
    # ax.plot(np.linspace(0, len(average_values), num = len(average_values)), np.full(shape = len(average_values), fill_value= -3*std_dev), color = 'lime') 
    fig.colorbar(image) 
    
    

    # plt.savefig(f"{log_path}/fig.pdf")
    plt.show()
    break

# with open(f"{log_path}/log.txt", 'x') as file:
#     file.write(f'Model: {model_name}\n')
#     file.write(f'Average predicted effect of each feature\n')
#     file.write(str(average_values))

