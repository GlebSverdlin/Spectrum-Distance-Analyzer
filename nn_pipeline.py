from network import *
import os
import datetime
from secret import *
print("Comment:")
comment = input()

network = start_network()

optimizer, loss_fn, epochs, train_dl, eval_dl = init_parameters(network, 5e-5, 30, 2000)

#ВОТ ЭТО МЕНЯТЬ ВРУЧНУЮ КАЖДЫЙ ЗАПУСК!!!
print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')
run = 'run4'
name = 'sann_v01'
date = str(datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
data_name = str(name+"_"+run+"_"+date)
log_path = str(logging)+str(data_name)
os.mkdir(log_path)

print(f"Run:{run}, model: {network}, logging to {log_path}.")

tr, itr = train_network(train_dl, network, loss_fn, optimizer, epochs)

with open(f"{log_path}/log.txt",'x') as file:
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

plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
plt.show()
plt.savefig(f"{log_path}/{data_name}.pdf")
print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')

eval_network(eval_dl, network, loss_fn)

print('ПРОВЕРИТЬ ОБНОВЛЕНИЕ ДАННЫХ ЛОГГИРОВАНИЯ')
