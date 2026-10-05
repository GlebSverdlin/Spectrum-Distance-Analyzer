from network import *
import os
import datetime
from secret import *
from scipy.signal import savgol_filter
import json

print("Comment:")
comment = input()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

network = start_network(device)

optimizer, loss_fn, epochs, train_dl, eval_dl = init_parameters(network, 1e-5, 120, 5)

#ВОТ ЭТО МЕНЯТЬ ВРУЧНУЮ КАЖДЫЙ ЗАПУСК!!!
print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')
run = 'run25'
date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+run+"_"+date)
log_path = os.path.join(logging, data_name)
os.mkdir(log_path)

print(f"Run:{run}, model: {network}, logging to {log_path}.")

tr, itr, model, times = train_network(train_dl, network, loss_fn, optimizer, epochs, device)


torch.save(model.state_dict(), PATH+name+run+date)


with open(f"{log_path}/log.json",'x') as file:
    json.dump({"comment" : comment, "network" : str(network), "optimizer" : str(optimizer), "loss_fn" : str(loss_fn), "times" : str(times), "losses" : tr}, file, indent = 4)


        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,7))

    tr_filter = savgol_filter(tr, window_length = int(len(tr)/10), polyorder = 3)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr_filter, color = 'r')
    plt.plot(1,1,color='w')
    plt.legend([str("Max: " + str(max)+'; Idx: '+str(max_idx)), str('Min: ' + str(min)+'; Idx: '+str(min_idx)),str('Avrg: '+ str(avrg))])
    plt.savefig(f"{log_path}/{data_name}.pdf")
    plt.show()


print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

# eval_network(eval_dl, network, loss_fn)

print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

def plot_losses(losses):
    for i in losses:
        if losses.index(i)%2==0:
            tr.append(i)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), savgol_filter(tr, window_length = len(tr)/100, polyorder = 3))
    plt.show()

