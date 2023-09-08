import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.model_selection import train_test_split





def jex_HAR_RV(stock1):
  

    vol = []

    sec = stock1['time_id']

    volatility1 = stock1['volatility']

    time_bucket = stock1['time_bucket']

    vol.append(pd.DataFrame({'time_id': sec, 'time_bucket': time_bucket, 'volatility': volatility1}))



    vol_train = []

    vol_test = []



    split_index = int(len(vol[0]) * 0.8)

    vol_train.append(vol[0][0:split_index])

    vol_test.append(vol[0][split_index:])





    len_train = len(vol_train[0]['volatility'])







    list_HAV_1 = []

    list_vol_1 = []

    list_HAV = []





    mean_vol_5 = pd.Series(vol_train[0]['volatility']).rolling(window=5, min_periods=0).mean()[22:]

    mean_vol_22 = pd.Series(vol_train[0]['volatility']).rolling(window=22, min_periods=0).mean()[22:]

    voll = vol_train[0]['volatility'][22:]

    list_HAV_1.append(pd.DataFrame({

        'vol': voll,

        'mean_vol_5': mean_vol_5.values,

        'mean_vol_22': mean_vol_22.values

    }))





    vol_1 = vol_train[0]['volatility'][21:-1]

    df = pd.DataFrame({'vol_1': vol_1.values})

    df.index = range(22, 22+len(df))

    list_vol_1.append(df)





    merged_df = pd.concat([list_HAV_1[0], list_vol_1[0]], axis=1)

    merged_df = merged_df[['vol', 'vol_1', 'mean_vol_5', 'mean_vol_22']]

    list_HAV.append(merged_df)



    list_HAV_2 = []

    list_vol_2 = []

    list_HAV_test = []



    mean_vol_5 = pd.Series(vol_test[0]['volatility']).rolling(window=5, min_periods=0).mean()[22:]

    mean_vol_22 = pd.Series(vol_test[0]['volatility']).rolling(window=22, min_periods=0).mean()[22:]

    voll = vol_test[0]['volatility'][22:]

    list_HAV_2.append(pd.DataFrame({

        'vol': voll,

        'mean_vol_5': mean_vol_5.values,

        'mean_vol_22': mean_vol_22.values

    }))



    vol_2 = vol_test[0]['volatility'][21:-1]

    df = pd.DataFrame({'vol_1': vol_2.values})

    df.index = range(split_index +22, split_index+len(df)+22)

    list_vol_2.append(df)



    merged_df = pd.concat([list_HAV_2[0], list_vol_2[0]], axis=1)

    merged_df = merged_df[['vol', 'vol_1', 'mean_vol_5', 'mean_vol_22']]

    list_HAV_test.append(merged_df)



    quar = []



    def comp_quar(x):

        return len(x) / 3 * sum([i**4 for i in x])



    df = vol[0]

    df_agg = df.groupby('time_bucket')['volatility'].agg(comp_quar).reset_index()

    df_agg.columns = ['time_bucket', 'quarticity']

    quar.append(df_agg)



    HAV_wls_models = []





    model = sm.formula.ols('vol ~ vol_1 + mean_vol_5 + mean_vol_22', data= list_HAV[0],

                            weights=list_HAV[0]['vol_1'] / np.sqrt(quar[0]['quarticity'][22:(len_train - 1)])).fit()

    HAV_wls_models.append(model)





    predictions = []



    newdata = pd.DataFrame({'vol_1': list_HAV_test[0]['vol'],

                            'mean_vol_5': list_HAV_test[0]['mean_vol_5'],

                            'mean_vol_22': list_HAV_test[0]['mean_vol_22']})

    predictions.append(HAV_wls_models[0].predict(newdata).tolist())





    df_list = []





    min_length = min(len(list_HAV_test[0]['vol']), len(predictions[0]))

    stock_pre = pd.DataFrame({'time_id': vol[0]['time_id'][split_index+22:],

                                'time_bucket': vol[0]['time_bucket'][split_index+22:],

                                'forecast': predictions[0][0:min_length]})

    df_list.append(stock_pre)

    combined_df = pd.concat(df_list, ignore_index=True)

    merged_df = pd.merge(stock1, combined_df, on=['time_id', 'time_bucket'], how='left')



    merged_df['forecast'].fillna(0, inplace=True)

    

    return merged_df



stock = pd.read_csv("/Users/jes/Downloads/output1_Jex_dataset_for_test.csv") 

print(jex_HAR_RV(stock))

print(stock)
