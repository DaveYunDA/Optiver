#This is a replication of the code to calculate volatility from ed1.ipynb so that it can be called as 1 single function
import pandas as pd
import numpy as np

def vol(stock1) -> pd.DataFrame:
    #inputs are stock 1

    log_r1 = []

    time_IDs = np.unique(stock1.iloc[:, 0])

    #Generate Weighted Average Price
    stock1["WAP"] = (stock1["bid_price1"] * stock1["ask_size1"] + stock1["ask_price1"] * stock1["bid_size1"]) / (stock1["bid_size1"] + stock1["ask_size1"])

    for i in range(len(time_IDs)):
        sec = stock1.loc[stock1.iloc[:, 0] == time_IDs[i], 'seconds_in_bucket'].values
        price = stock1.loc[stock1.iloc[:, 0] == time_IDs[i], 'WAP'].values
        log_r = np.log(price[1:] / price[0:(len(price) - 1)])
        log_r1.append(pd.DataFrame({'time': sec[1:], 'log_return': log_r}))
        time_no_change = np.setdiff1d(np.arange(1, 601), log_r1[i]['time'].values)
        if len(time_no_change) > 0:
            new_df = pd.DataFrame({'time': time_no_change, 'log_return': 0})
            log_r1[i] = pd.concat([log_r1[i], new_df])
            log_r1[i] = log_r1[i].sort_values(by='time')

    vol = []
    def comp_vol(x):
        return np.sqrt(np.sum(x ** 2))

    for i in range(len(log_r1)):
        log_r1[i]['time_bucket'] = np.ceil(log_r1[i]['time'] / 30)
        vol.append(log_r1[i].groupby('time_bucket')['log_return'].agg(comp_vol).reset_index())
        vol[i].columns = ['time_bucket', 'volatility']

    #Format output as an array instead of a list
    #Crush all the sub-dataframes (one dataframe per 10 min chunk) into 1 array
    out = []
    for i in range(len(vol)):
        for j in range(20):
            out.append(vol[i]["volatility"][j])
    vol = np.array(out)


    return vol








