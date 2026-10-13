from network import *
import os
import datetime
from secret import *
from scipy.signal import savgol_filter
import json
import sys
import getopt
# print("Comment:")
# comment = input()
args = sys.argv[1:]
output_freq = 1
option = "r:"
long_option = ["reduced"]

args, vals = getopt.getopt(args,option, long_option)

try:
    for arg, val in args:
        if arg in ("-r", "--reduced"):
            output_freq = int(sys.argv[2])

except: print("ERR")

comment = '---now-commented---'
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

network = start_network(device)

optimizer, loss_fn, epochs, train_dl, eval_dl, threshold, patience, patience_bool = init_parameters(network, 1e-5, 1600, 100, 0.7, 3, True) #model, LR, batch, epochs, threshold, patience, patience_on

#ВОТ ЭТО МЕНЯТЬ ВРУЧНУЮ КАЖДЫЙ ЗАПУСК!!!
print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')
run = 'run25'
date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+run+"_"+date)
log_path = os.path.join(logging, data_name)
os.mkdir(log_path)

print(f"Run:{run}, model: {network}, logging to {log_path}.")

model_save_path = os.path.join(PATH, name+run+date)
os.mkdir(model_save_path)



tr, itr, model, times = train_network(train_dl, network, loss_fn, optimizer, epochs, device, threshold, model_save_path, patience, patience_bool, output_freq)


torch.save(model.state_dict(), model_save_path+'/'+name+run+date)


with open(f"{log_path}/log.json",'x') as file:
    json.dump({"comment" : comment, "network" : str(network), "optimizer" : str(optimizer), "loss_fn" : str(loss_fn), "times" : str(times), "losses" : tr}, file, indent = 4)

    plt.style.use('bmh')

    max = np.max(tr)
    min = np.min(tr)
    avrg = np.average(tr)
    max_idx = tr.index(max)
    min_idx = tr.index(min)
    median = np.median(tr)
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,7))

    tr_filter = savgol_filter(tr, window_length = int(len(tr)/10), polyorder = 3)

    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr_filter, color = 'r')
    plt.plot(1,1,color='w')
    plt.plot(1,1,color='w')
    plt.legend([str("Max: " + str(max)+'; Idx: '+str(max_idx)), str('Min: ' + str(min)+'; Idx: '+str(min_idx)),str('Avrg: '+ str(avrg)), str('Mdn: '+str(median))])
    plt.savefig(f"{log_path}/{data_name}.pdf")
    plt.show()



print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

# eval_network(eval_dl, network, loss_fn)

# print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

# def plot_losses(losses):
#     for i in losses:
#         if losses.index(i)%2==0:
#             tr.append(i)
#     plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
#     plt.plot(np.linspace(0, len(tr), num = len(tr)), savgol_filter(tr, window_length = len(tr)/100, polyorder = 3))
#     plt.show()

