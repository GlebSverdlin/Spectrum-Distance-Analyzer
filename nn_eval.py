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

name = 'sann_v01_2'
date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+date)
log_path = str(eval_logging)+str(data_name)
os.mkdir(log_path)

eval_data = SpectralDataset('aspcap', 'eval')

model = NeuralNetwork()
model.load_state_dict(torch.load(path, weights_only = True))
model.eval()

eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False)

with torch.no_grad():
    answers = []
    corrects = []

    probabilities = []

    prediction_number = len(eval_data)
    for iter, (features, label) in enumerate(eval_dataloader):
        while iter < prediction_number:
            probability = model(features)
            correct = label
            
            if correct == 1:
                probabilities.append(1-probability)
            else: probabilities.append(probability)

            print(f'Iteration #{iter}:')
            print(f'Label: {label}')
            print(f'Model probability: {probability}')        
            corrects.append(correct[0])
            answers.append(probability[0])           
         
            break

average = np.average(probabilities)

print(f'Average error: {average}')

plt.plot(np.linspace(0, len(answers), num = len(answers)), corrects, 'x')
plt.plot(np.linspace(0, len(answers), num = len(answers)), answers, 'rx')

text = []

for i in range(len(answers)):
    text.append(corrects[i].item())
    text.append(answers[i].item())

with open(f"{log_path}/log.txt", 'x') as file:
    file.write(f'Tested model: {model_name}\n')
    file.write(f'Answers:')
    file.write(f'{str(text)}')
plt.show()

plt.savefig(f"{log_path}/fig.pdf")



