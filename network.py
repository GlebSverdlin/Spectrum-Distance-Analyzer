import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import *
import matplotlib.pyplot as plt
import numpy as np 
from secret import *
import time

train_data = SpectralDataset('aspcap_nz', 'train')
eval_data = SpectralDataset('aspcap_nz', 'eval')
test_data = SpectralDataset('aspcap_nz', 'test')

class NeuralNetwork(nn.Module):
        def __init__(self):
                super().__init__()
                self.neural_stack=nn.Sequential(
            #=======MODEL_1========
                        nn.Linear(7514, 4096),
                        nn.ReLU(),
                        nn.Linear(4096, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 50),
                        nn.ReLU(),
                        nn.Linear(50,30),
                        nn.ReLU(),
                        nn.Linear(30, 5),
                        nn.ReLU(),
                        nn.Linear(5,1),
                        nn.Sigmoid()
                        #
            #=======MODEL_2========
                        # nn.Linear(7514, 1072),
                        # nn.LeakyReLU(),
                        # nn.Linear(1072, 1072),
                        # nn.LeakyReLU(),
                        # nn.Linear(1072, 1072),
                        # nn.LeakyReLU(),
                        # nn.Linear(1072, 1072),
                        # nn.LeakyReLU(),
                        # nn.Linear(1072, 1072),
                        # nn.LeakyReLU(),
                        # nn.Linear(1072, 1072),
                        # nn.LeakyReLU(),
                        # nn.Linear(1072, 268),
                        # nn.LeakyReLU(),
                        # nn.Linear(268, 268),
                        # nn.LeakyReLU(),
                        # nn.Linear(268,67),
                        # nn.LeakyReLU(),
                        # nn.Linear(67, 24),
                        # nn.LeakyReLU(),
                        # nn.Linear(24,1),
                        # nn.Sigmoid() 

            #=======MODEL_3========
                        # nn.Linear(7514, 500),
                        # nn.LeakyReLU(),
                        # nn.Linear(500, 300),
                        # nn.LeakyReLU(),
                        # nn.Linear(300, 200),
                        # nn.LeakyReLU(),
                        # nn.Linear(200, 100),
                        # nn.LeakyReLU(),
                        # nn.Linear(100,50),
                        # nn.LeakyReLU(),
                        # nn.Linear(50,25),
                        # nn.LeakyReLU(),
                        # nn.Linear(25,1),
                        # nn.Sigmoid()

            #=======MODEL_4========
                        # nn.Linear(7514, 256),
                        # # nn.LeakyReLU(),
                        # # nn.Linear(2505,835),
                        # # nn.LeakyReLU(),
                        # # nn.Linear(835, 256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,256),
                        # nn.LeakyReLU(),
                        # nn.Linear(256,128),
                        # nn.LeakyReLU(),
                        # nn.Linear(128,64),
                        # nn.LeakyReLU(),
                        # nn.Linear(64,32),
                        # nn.LeakyReLU(),
                        # nn.Linear(32,16),
                        # nn.LeakyReLU(),
                        # nn.Linear(16,1),
                        # nn.Sigmoid()


                )
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

def start_network(device):
        network = NeuralNetwork()
        network.to(device)
        network.apply(init_weights)
        print('Network:')
        print(network)
        return network

def init_parameters(model, learn_rate, batch, epochs, threshold, patience, patience_bool):
        optimizer = optim.Adam(model.parameters(), lr = learn_rate)
        # optimizer = optim.SGD(model.parameters(), lr = learn_rate)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss = nn.BCELoss()
        train_dataloader = DataLoader(train_data, batch_size=batch, shuffle = True)
        eval_dataloader = DataLoader(eval_data, batch_size=len(eval_data), shuffle = False)
        test_dataloader = DataLoader(test_data, batch_size=batch, shuffle = False)

        print(f"Optimizer: {optimizer}")
        print(f"Loss function: {loss}")
        print(f"Batch size: {batch}")
        print("Loaders:") 
        print(train_dataloader, len(train_dataloader))
        print(eval_dataloader)
        print(f"Epochs: {epochs}")

        return optimizer, loss, epochs, train_dataloader, eval_dataloader, test_dataloader, threshold, patience, patience_bool


def train_network(train_loader, test_loader, model, loss_fn, optimizer, epochs, device, threshold, model_save_path, patience, patience_bool, output_freq):
        losses = []
        test_losses = []
        iters = []
        times = []
        step_num = 0
        patience_on = patience_bool
        threshold_passed = False
        dynamic = os.path.join(model_save_path, 'dynamic')
        os.mkdir(dynamic)
        dyn_counter = 0
        for epoch in range(epochs):
                # size = len(loader.dataset)
                model.train()
                train_epoch_start = time.perf_counter()
                for iter, (features, label) in enumerate(test_loader, start = 0):
                        # print(f"Iter: {iter}\n Features: {features}\n Label: {label}\n\n")
                        train_step_start = time.perf_counter()
                        batch = len(label)
                        prediction = model(features.to(device))
                        train_loss = loss_fn(prediction, label.to(device))

                        train_loss.backward()
                        optimizer.step() 
                        optimizer.zero_grad()
                        train_step_end = time.perf_counter()
                        train_step_dur = train_step_end - train_step_start                        
                        # print (f"\tIter {iter}\n\n")
                
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for iter, (features, label) in enumerate(train_loader):
                        prediction = model(features.to(device))
                        test_loss = loss_fn(prediction, label.to(device))
                

                end = time.time()
                step_num+=1
                train_loss = train_loss.item()
                losses.append(train_loss)
                test_losses.append(test_loss)
                iters.append(step_num)
                times.append(train_step_dur) 

                if train_loss<threshold:
                        dyn_counter+=1
                        folder = os.scandir(dynamic)
                        for item in folder:
                                os.remove(os.path.join(dynamic, item.name))
                        torch.save(model.state_dict(), dynamic+'/'+f'dynamic_{dyn_counter}_epoch{epoch}')
                        threshold = train_loss
                        threshold_passed = True
                else:
                        if patience_on and threshold_passed:
                            patience-=1

                if patience == 0:
                        print(f"=========Training ended due to patience 0=========")
                        break

                train_epoch_end = time.perf_counter()
                if epoch%output_freq == 0:
                        print(f"iteration: {step_num}")
                        print(f"Train loss: {train_loss:>7f}")
                        print(f"Test loss: {test_loss:>7f}")
                        print (f"Loaders num: {len(train_loader)}\n")
                        print(f"Duration step {iter}: {train_step_dur}")
                        print(f"Epoch {epoch} duration {train_epoch_end - train_epoch_start} last iter {iter}")
                        print(f"ETA: {(train_epoch_end - train_epoch_start)*(epochs - epoch)} s")



        return losses, test_losses, iters, model, times

                                       
# def eval_network(loader, model, loss_fn):
#     model.eval()
#     size = len(loader.dataset)
#     num_batches = len(loader)
#     test_loss, correct = 0, 0
# #TODO:percentage calculation rework
#     with torch.no_grad():
#         for features, labels in loader:
#             predictions = model(features)
#             test_loss += loss_fn(predictions, labels).item()
#             correct += (predictions.argmax(1)==labels).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#


