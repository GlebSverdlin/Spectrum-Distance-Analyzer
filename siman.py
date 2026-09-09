import random
import math
import numpy.random as nprnd
import numpy as np
import torch
import torch.optim as optim

class SimulatedAnnealing(optim.Optimizer):
    def __init__(self, lr, temp_start, cool_rate):
        self.lr = lr
        self.temp_start = temp_start
        self.cool_rate = cool_rate
        self.param_groups = []

    
    
    def step(self, model, loss_fn, features, label): 
        update_flag = False

        on_step_params = model.parameters()
        on_step_loss = loss_fn(model(features), label)

        for item in model.parameters():
            delta = torch.rand(item.shape)#-0.5*torch.ones(item.shape)
            item = item + delta *10*self.lr*np.log(self.temp_start)

        prediction = model(features)

        after_step_loss = loss_fn(prediction, label)
        
        loss = after_step_loss
        update_flag = True

        if ((after_step_loss>on_step_loss) and (torch.Tensor(1).uniform_() > ((after_step_loss-on_step_loss)/self.temp_start).exp())):
            loss = on_step_loss
            for iter, item in enumerate(model.parameters()):
                item.data = on_step_params[iter].data
            update_flag = False
        self.temp_start/=self.cool_rate

        return loss, update_flag
