import torch
from torch import nn 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import test_dataset
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import random

training_data = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/training/training_set.csv"
testing_data = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/testing/testing_set.csv"
validating_data = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/validation/validating_set.csv"
training_labels = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/training/training_labels.csv"
testing_labels = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/testing/testing_labels.csv"
validating_labels = "/home/gleb/Programming/Python/SpectrumAnalyzer/test_dataset/validation/validating_labels.csv"

batch_size = 128

train_dataloader = DataLoader(test_dataset.training_dataset(training_data, training_labels), batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset.testing_dataset(training_data, training_labels), batch_size=1, shuffle=True)
validation_dataloader = DataLoader(test_dataset.validating_dataset(training_data, training_labels), batch_size, shuffle=True)

device = "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.stack = nn.Sequential(
        nn.Linear(6, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.ReLU(),
        nn.Linear(2,1)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.stack(x)
        return x


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()

        train_iterator = iter(train_dataloader)
        valid_iterator = iter(test_dataloader)
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            print(np.shape(inputs))
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_iterator)
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
            target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(valid_iterator)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f},accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))

def predict(model, test_loader, sample_index): 
    for batch in test_loader:
        input, label = batch
        output = model(input)
        print(f'Prediction: ', {output}, 'expected: ', {label})

test_network = NeuralNetwork()
optimizer = optim.Adam(test_network.parameters(), lr=0.001)


