from network import *
import os
import datetime
from secret import *
from scipy.signal import savgol_filter
print("Comment:")
comment = input()

network = start_network()

optimizer, loss_fn, epochs, train_dl, eval_dl = init_parameters(network, 1e-5, 30, 1000)

#ВОТ ЭТО МЕНЯТЬ ВРУЧНУЮ КАЖДЫЙ ЗАПУСК!!!
print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')
run = 'run22'
name = 'sann_v01_2'
date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+run+"_"+date)
log_path = str(logging)+str(data_name)
os.mkdir(log_path)

print(f"Run:{run}, model: {network}, logging to {log_path}.")

tr, itr, model = train_network(train_dl, network, loss_fn, optimizer, epochs)


torch.save(model.state_dict(), PATH+name+run+date)

with open(f" {log_path}/log.txt",'x') as file:
    file.write("COMMENT:\n")
    file.write(comment)
    file.write("\n=================MODEL=================\n")
    file.write(str(network))
    file.write("\n================OPTIMIZER================\n")
    file.write(str(optimizer))
    file.write("\n================LOSS FUNC================\n")
    file.write(str(loss_fn))
    file.write("\n================LOSSES================\n")
    iterations = []
    for i in range(len(tr)):
        iterations.append(itr[i])
        iterations.append(tr[i])

    file.write(str(iterations))

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

    plt.savefig(f"{log_path}/{data_name}.pdf")
print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

eval_network(eval_dl, network, loss_fn)

print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

def plot_losses(losses):
    for i in losses:
        if losses.index(i)%2==0:
            tr.append(i)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
    plt.plot(np.linspace(0, len(tr), num = len(tr)), savgol_filter(tr, window_length = len(tr)/100, polyorder = 3))
    plt.show()

