import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import *
import matplotlib.pyplot as plt
import numpy as np 

name = 'sann_v01' 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

train_data = SpectralDataset('aspcap', 'train')
eval_data = SpectralDataset('aspcap', 'eval')
'''
n=126
plt.figure(figsize=(23,5))

train_data_length = len(train_data[1][0])
print (f'train_data_length: {train_data_length}')
print (f'eval_data[n]: {eval_data[n]}')

print(eval_data[n][0])

plt.plot(np.linspace(0, train_data_length, num = train_data_length), eval_data[n][0])
plt.show()
'''
class NeuralNetwork(nn.Module):
        def __init__(self):
                super().__init__()
                self.neural_stack=nn.Sequential(
                        nn.Linear(8575, 4096),
                        nn.LeakyReLU(),
                        nn.Linear(4096, 2048),
                        nn.LeakyReLU(),
                        nn.Linear(2048, 1024),
                        nn.LeakyReLU(),
                        nn.Linear(1024, 512),
                        nn.LeakyReLU(),
                        nn.Linear(512, 128),
                        nn.LeakyReLU(),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(),
                        nn.Linear(128, 50),
                        nn.LeakyReLU(),
                        nn.Linear(50,30),
                        nn.LeakyReLU(),
                        nn.Linear(30, 5),
                        nn.LeakyReLU(),
                        nn.Linear(5,1),
                        nn.Sigmoid() 
                )
                #TODO: implement conversion to float32
                self.double()
        def forward(self, features):
                prediction = self.neural_stack(features)
                return prediction

def init_weights(layer):
        if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                layer.bias.data.fill_(0.01)

def print_weights(model, msg):
    print("\n", msg)
    for name, p in model.named_parameters():
        print(
            name,
            "mean =", p.data.mean().item(),
            "max =", p.data.abs().max().item(),
        )

def start_network():
        network = NeuralNetwork()
        network.apply(init_weights)
        network.to(device)
        print('Network:')
        print(network)
        return network

def init_parameters(model, learn_rate, batch, epochs):
        optimizer = optim.Adam(model.parameters(), lr = learn_rate)
        # optimizer = optim.SGD(model.parameters(), lr = learn_rate)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss = nn.BCELoss()
        train_dataloader = DataLoader(train_data, batch_size=batch, shuffle = True)
        eval_dataloader = DataLoader(eval_data, batch_size=batch, shuffle=False)

        print(f"Optimizer: {optimizer}")
        print(f"Loss function: {loss}")
        print("Loaders:")
        print(train_dataloader, len(train_dataloader))
        print(eval_dataloader)
        print(f"Epochs: {epochs}")

        return optimizer, loss, epochs, train_dataloader, eval_dataloader


def train_network(loader, model, loss_fn, optimizer, epochs):
        losses = []
        for epoch in range(epochs):
                size = len(loader.dataset)
                model.train()
                for iter, (features, label) in enumerate(loader,0):
                        batch = len(label)
                        prediction = model(features)
                        loss = loss_fn(prediction, label)

                        loss.backward()
                        optimizer.step() 
                        optimizer.zero_grad()

                                          
                        if iter % 100 == 0:
                            loss = loss.item()
                            print(f"loss: {loss:>7f}")
                            losses.append(loss)
        torch.save(model.state_dict(), PATH+name)
        return losses

                                       
def eval_network(loader, model, loss_fn):
    model.eval()
    size = len(loader.dataset)
    num_batches = len(loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features)
            test_loss += loss_fn(predictions, labels).item()
            correct += (predictions.argmax(1)==labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




