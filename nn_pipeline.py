from network import *

network = start_network()

optimizer, loss_fn, epochs, train_dl, eval_dl = init_parameters(network, 1e-4, 30, 100)

tr = train_network(train_dl, network, loss_fn, optimizer, epochs)

plt.plot(np.linspace(0, len(tr), num = len(tr)), tr)
plt.show()

eval_network(eval_dl, network, loss_fn)

