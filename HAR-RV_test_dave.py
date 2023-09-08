import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def get_volatility_by_time_id(stock_data):
    vol = []
    time_IDs = stock_data['time_id'].unique()

    for i in range(len(time_IDs)):
        sec = stock_data.loc[stock_data['time_id'] == time_IDs[i], 'time_bucket'].tolist()
        volatility1 = stock_data.loc[stock_data['time_id'] == time_IDs[i], 'volatility_stock_0'].tolist()
        time_id = stock_data.loc[stock_data['time_id'] == time_IDs[i], 'time_id'].iloc[0]
        vol.append(pd.DataFrame({'time_bucket': sec, 'volatility': volatility1, 'time_id': time_id}))
    
    return pd.concat(vol)



def split_volatility_by_time_id(volatility_list, train_size=480):
    vol_train = []
    vol_test = []

    for i in range(len(volatility_list)):
        vol_train.append(volatility_list[i][0:train_size])
        vol_test.append(volatility_list[i][train_size:])

    train_data = pd.concat(vol_train)
    test_data = pd.concat(vol_test)
    
    return train_data, test_data


def compute_list_HAV(vol_train, vol_test):
    def compute_HAV(df):
        mean_vol_5 = pd.Series(df['volatility']).rolling(window=5, min_periods=0).mean()[22:]
        mean_vol_22 = pd.Series(df['volatility']).rolling(window=22, min_periods=0).mean()[22:]
        voll = df['volatility'][22:]
        df_HAV = pd.DataFrame({
            'vol': voll,
            'mean_vol_5': mean_vol_5.values,
            'mean_vol_22': mean_vol_22.values
        })
        return df_HAV

    def compute_vol_1(df):
        vol_1 = df['volatility'][21:-1]
        df_vol_1 = pd.DataFrame({'vol_1': vol_1.values})
        df_vol_1.index = range(22, 22+len(df_vol_1))
        return df_vol_1
    
    list_HAV = []
    list_vol_1 = []
    for i in range(len(vol_train)):
        merged_df = pd.concat([compute_HAV(vol_train[i]), compute_vol_1(vol_train[i])], axis=1)
        merged_df = merged_df[['vol', 'vol_1', 'mean_vol_5', 'mean_vol_22']]
        list_HAV.append(merged_df)

    list_HAV_test = []
    list_vol_2 = []
    for i in range(len(vol_test)):
        merged_df = pd.concat([compute_HAV(vol_test[i]), compute_vol_1(vol_test[i])], axis=1)
        merged_df = merged_df[['vol', 'vol_1', 'mean_vol_5', 'mean_vol_22']]
        list_HAV_test.append(merged_df)

    return list_HAV, list_HAV_test



def compute_predictions(vol, list_HAV, quar, len_train, list_HAV_test):
    HAV_wls_models = []
    def comp_quar(x):
        return len(x) / 3 * sum([i**4 for i in x])

    for i in range(len(vol)):
        df = vol[i]
        df_agg = df.groupby('time_bucket')['volatility'].agg(comp_quar).reset_index()
        df_agg.columns = ['time_bucket', 'quarticity']
        quar.append(df_agg)

    for i in range(len(vol)):
        model = sm.formula.ols('vol ~ vol_1 + mean_vol_5 + mean_vol_22', data=list_HAV[i],
                               weights=list_HAV[i]['vol_1'] / np.sqrt(quar[i]['quarticity'][22:(len_train - 1)])).fit()
        HAV_wls_models.append(model)

    predictions = []
    for i in range(len(HAV_wls_models)):
        newdata = pd.DataFrame({'vol_1': list_HAV_test[i]['vol'],
                                'mean_vol_5': list_HAV_test[i]['mean_vol_5'],
                                'mean_vol_22': list_HAV_test[i]['mean_vol_22']})
        predictions.append(HAV_wls_models[i].predict(newdata).tolist())
    
    return predictions



def predict_volatility(stock_data, list_HAV_test, predictions):
    df_list = []
    for i in range(len(list_HAV_test)):
        min_length = min(len(list_HAV_test[i]['vol']), len(predictions[i]))
        stock_pre = pd.DataFrame({'time_id': stock_data[i]['time_id'][502:600],
                                  'time_bucket': stock_data[i]['time_bucket'][502:600],
                                  'predicted_volatility': predictions[i][0:min_length]})
        df_list.append(stock_pre)

    combined_df = pd.concat(df_list, ignore_index=True)

    merged_df = pd.merge(stock_data[0], combined_df, on=['time_id', 'time_bucket'], how='left')
    
    return merged_df
