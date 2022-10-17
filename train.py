
import imp
import time
import argparse
from sklearn.preprocessing import MinMaxScaler
from sympy import im
import pandas as pd
import torch
import numpy as np

from cfg import parse_cfg
from DataUtil import getStockData
from DataUtil import plotThestocks
from DataUtil import normalizeData
from DataUtil import split_data
from trainModelGRU import trainModelGRU
from plotPredictions import plotPredictions

def main():
    tickerCode    = FLAGS.ticker
    cfgfile = FLAGS.config
    
    #print(ticker)

    net_options   = parse_cfg(cfgfile)[0]    
    # print(net_options)
    

    lookback   = int(net_options['lookback'])
    input_dim  = int(net_options['input_dim'])
    hidden_dim = int(net_options['hidden_dim'])
    num_layers = int(net_options['num_layers'])
    output_dim = int(net_options['output_dim'])
    num_epochs = int(net_options['num_epochs'])


    data = getStockData(tickerCode)
    print(data.head())

    print(data.shape)
    
    # plotThestocks(data)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # price = normalizeData(scaler, data)
    price = data[['Close']]
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    
    
    x_train, y_train_gru, x_test, y_test = split_data(price, lookback)

    
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train_gru.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)
    
    y_train_pred, model = trainModelGRU(input_dim, hidden_dim
        , output_dim, num_layers, num_epochs, x_train, y_train_gru, trainModelGRU)

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))

# #     plotPredictions(original, predict)
    
    
    y_train_pred, y_test_pred = invertPrediction(y_train_pred, x_test, model)

# invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())


    original = scaler.inverse_transform(price['Close'].values.reshape(-1,1))

    plotPredictions(y_train_pred, y_test_pred, price, original, lookback)

    predictions = predictNextTenDays(price.tail(30), model)
    predict_next_10 = scaler.inverse_transform(predictions)
    print("price of "+tickerCode+ " for the next 10 days:")
    print(predict_next_10)
# #     print(data.head())
    

def predictNextTenDays(price, model):
    x_test_new = (price).to_numpy()
    x_test_new = x_test_new.reshape((1, x_test_new.shape[0], 1))
    # y_test = data[train_set_size:,-1,:]

    x_test_tens = torch.from_numpy(x_test_new).type(torch.Tensor)
    x_test_tens.size()

    # make predictions
    pred = []
    for i in range(1, 10):
        y_test_pred_new = model(x_test_tens)
        
        x_test_new = np.append(x_test_new, y_test_pred_new.detach().numpy())
        x_test_new = x_test_new.reshape((1, x_test_new.shape[0], 1))
        x_test_new = x_test_new[:, 1:, :]
        x_test_tens = torch.from_numpy(x_test_new).type(torch.Tensor)
    #     print(x_test_tens)
        
        y_test_pred_new = y_test_pred_new.detach().numpy()
        # print(y_test_pred_new)
        pred.append([y_test_pred_new[0][0]])
        
    return pred

def invertPrediction(y_train_pred, x_test, model):
    # make predictions
    y_test_pred = model(x_test)
    
    return [y_train_pred, y_test_pred]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ticker', '-t',
        type=str, default='GOOGL', help='Enter Yahoo Finance Ticker of the stock')

    parser.add_argument('--config', '-c',
        type=str, default='stk.cfg', help='configuration file')
        

    FLAGS, _ = parser.parse_known_args()

    main()





