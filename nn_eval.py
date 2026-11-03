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
import json
from matplotlib.widgets import LassoSelector
import matplotlib as mpl

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

eval_data = SpectralDataset('research_tp', 'res')
eval_data2 = SpectralDataset('research_tn', 'res')
eval_data.concatenate(eval_data2)
model = NeuralNetwork()
model.load_state_dict(torch.load(path, weights_only = True))


date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+date)
log_path = os.path.join(eval_logging, str(data_name+'-test_date-'+date))
os.mkdir(log_path)
eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False)
model.eval()

with torch.no_grad():
    answers = []
    corrects = []

    errors = []

    prediction_number = len(eval_data)
    for iter, (features, label) in enumerate(eval_dataloader):
        while iter < prediction_number:
            probability = model(features)
            correct = label
            
            if correct == 1:
                errors.append(1-probability)
            else: errors.append(probability)

            print(f'Iteration #{iter}:')
            print(f'Label: {label}')
            print(f'Model probability: {probability}')        
            corrects.append(correct[0])
            answers.append(probability[0])           
        
            break

average = np.average(errors)

print(f'Average error: {average}')

plt.figure(figsize=(25,10))
plt.style.use('bmh')
plt.plot(np.linspace(0, len(answers), num = len(answers)), corrects, '_')
plt.axhline(y=0.25, color = 'r', linestyle = '-')
plt.axhline(y=0.75, color = 'r', linestyle = '-')
plt.scatter(np.linspace(0, len(answers), num = len(answers)), answers, c = errors, cmap = 'plasma', marker='x')
plt.colorbar()
plt.legend(['Average: '+str(average)])
plt.savefig(f"{log_path}/fig.pdf")

cor = []
ans = []
err = []

for i in range(len(answers)):
    cor.append(corrects[i].item())
    ans.append(answers[i].item())
    err.append(errors[i].item())

with open(f"{log_path}/log.json", 'x') as file:
    json.dump({"network" : model_name, "answers" : ans, "corrects" : cor, "errors" : err}, file, indent = 4)

plt.show()




