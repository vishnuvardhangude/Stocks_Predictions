import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
import numpy as np

#get stock data
def getStockData(stCode, period="max"):
    msft = yf.Ticker(stCode)
    # get historical market data
    cdsl = msft.history(period)
    cdsl = cdsl.reset_index(level=0)

    del cdsl['Dividends']
    del cdsl['Stock Splits']

    return cdsl

def plotThestocks(data):
    #seaborn.set_style:{darkgrid, whitegrid, dark, white, ticks}
    sns.set_style("darkgrid")
    plt.figure(figsize = (15,9))
    plt.plot(data[['Close']])
    plt.xticks(range(0,data.shape[0],250),data['Date'].loc[::250],rotation=45)
    plt.title("Google Stock Price",fontsize=18, fontweight='bold')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price (USD)',fontsize=18)
    plt.show()



def normalizeData(scaler, data):
    price = data[['Close']]
#     price.info()
    
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    
    return price



def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.01*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)


    return [x_train, y_train_gru, x_test, y_test_gru]