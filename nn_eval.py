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

purpose = ''
args = sys.argv[1:]
option = "xem:"
long_option = ["explain, evaluate, model"]

args, vals = getopt.getopt(args,option, long_option)

try:
    for arg, val in args:
        if arg in ("-m", "--model"):
            model_name = sys.argv[2]
            print(model_name)
            path = PATH+model_name

        if arg in ('-x', '--explain'):
            purpose = 'x'
        if arg in ('e', '--evaluate'):
            purpose = 'e'

except: print(str(getopt.error))

print(purpose)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

eval_data = SpectralDataset('aspcap', 'eval')

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
# plt.plot(np.linspace(0, len(answers), num = len(answers)), corrects, '_')
# plt.scatter(np.linspace(0, len(answers), num = len(answers)), answers, c = errors, cmap = 'plasma', marker='x')
# plt.colorbar()
# plt.legend(['Average: '+str(average)])
# plt.savefig(f"{log_path}/fig.pdf")
text = []

for i in range(len(answers)):
    text.append(corrects[i].item())
    text.append(answers[i].item())

with open(f"{log_path}/log.txt", 'x') as file:
    file.write(f'Tested model: {model_name}\n')
    file.write(f'Answers:')
    file.write(f'{str(text)}')
# plt.show()



log_path = os.path.join(logs_shap, str(data_name+'-test_date-'+date))
os.mkdir(log_path)
eval_dataloader = DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
for iter, batch in enumerate(eval_dataloader):
    # data, _ = batch
    background_data = batch[0][:int(0.8*len(eval_data))]
    test_data = batch[0][int(0.8*len(eval_data)):]
    explainer = shap.DeepExplainer(model, background_data)
    shap_vals = explainer.shap_values(test_data)
    print(shap_vals[0][400:410])
    c = []
    for i in shap_vals[0]:
        c.append(10**(0+i))
    plt.scatter(np.linspace(0, len(shap_vals[0]), num = len(shap_vals[0])), shap_vals[0], c = c, cmap = 'plasma', marker = 'x')
    plt.colorbar()
    plt.show()
    break

