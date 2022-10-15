import numpy as np
import torch
import time
import pandas as pd

from GRU import *

def trainModelGRU(input_dim, hidden_dim, output_dim, num_layers, num_epochs
                  , x_train, y_train_gru, scaler):
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    gru = []
    
    for t in range(num_epochs):
        y_train_pred = model(x_train)
    
        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
    
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    training_time = time.time()-start_time    
    print("Training time: {}".format(training_time))
    
    return [y_train_pred, model]
