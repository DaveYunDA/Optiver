import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def dave_HAR_RV(stock):

    vol = []
    time_IDs = stock1['time_id'].unique()

    for i in range(len(time_IDs)):
        sec = stock1.loc[stock1['time_id'] == time_IDs[i], 'time_bucket'].tolist()
        volatility1 = stock1.loc[stock1['time_id'] == time_IDs[i], 'volatility'].tolist()
        time_id = stock1.loc[stock1['time_id'] == time_IDs[i], 'time_id'].iloc[0]
        vol.append(pd.DataFrame({'time_bucket': sec, 'volatility': volatility1, 'time_id': time_id}))


    vol_train = []
    vol_test = []

    for i in range(len(time_IDs)):
        vol_train.append(vol[i][0:480])
        vol_test.append(vol[i][480:])

    len_train = len(vol_train[0]['volatility'])
    list_HAV_1 = []
    list_vol_1 = []
    list_HAV = []

    for i in range(len(vol_train)):
        mean_vol_5 = pd.Series(vol_train[i]['volatility']).rolling(window=5, min_periods=0).mean()[22:]
        mean_vol_22 = pd.Series(vol_train[i]['volatility']).rolling(window=22, min_periods=0).mean()[22:]
        voll = vol_train[i]['volatility'][22:]
        list_HAV_1.append(pd.DataFrame({
            'vol': voll,
            'mean_vol_5': mean_vol_5.values,
            'mean_vol_22': mean_vol_22.values
        }))

    for i in range(len(vol_train)):
        vol_1 = vol_train[i]['volatility'][21:-1]
        df = pd.DataFrame({'vol_1': vol_1.values})
        df.index = range(22, 22+len(df))
        list_vol_1.append(df)

    for i in range(len(list_HAV_1)):
        merged_df = pd.concat([list_HAV_1[i], list_vol_1[i]], axis=1)
        merged_df = merged_df[['vol', 'vol_1', 'mean_vol_5', 'mean_vol_22']]
        list_HAV.append(merged_df)

    list_HAV_2 = []

    for i in range(len(vol_test)):
        mean_vol_5 = pd.Series(vol_test[i]['volatility']).rolling(window=5, min_periods=0).mean()[22:]
        mean_vol_22 = pd.Series(vol_test[i]['volatility']).rolling(window=22, min_periods=0).mean()[22:]
        voll = vol_test[i]['volatility'][22:]
        list_HAV_2.append(pd.DataFrame({
            'vol': voll,
            'mean_vol_5': mean_vol_5.values,
            'mean_vol_22': mean_vol_22.values
        }))

    list_vol_2 = []

    for i in range(len(vol_test)):
        vol_2 = vol_test[i]['volatility'][21:-1]
        df = pd.DataFrame({'vol_1': vol_2.values})
        df.index = range(502, 502+len(df))
        list_vol_2.append(df)

    list_HAV_test = []

    for i in range(len(list_HAV_2)):
        merged_df = pd.concat([list_HAV_2[i], list_vol_2[i]], axis=1)
        merged_df = merged_df[['vol', 'vol_1', 'mean_vol_5', 'mean_vol_22']]
        list_HAV_test.append(merged_df)


    quar = []
    def comp_quar(x):
        return len(x) / 3 * sum([i**4 for i in x])

    for i in range(len(vol)):
        df = vol[i]
        df_agg = df.groupby('time_bucket')['volatility'].agg(comp_quar).reset_index()
        df_agg.columns = ['time_bucket', 'quarticity']
        quar.append(df_agg)

    HAV_wls_models = []
    import statsmodels.api as sm

    for i in range(len(vol)):
        model = sm.formula.ols('vol ~ vol_1 + mean_vol_5 + mean_vol_22', data= list_HAV [i],
                    weights=list_HAV[i]['vol_1'] / np.sqrt(quar[i]['quarticity'][22:(len_train - 1)])).fit()
        HAV_wls_models.append(model)

    predictions = []

    for i in range(len(HAV_wls_models)):
        newdata = pd.DataFrame({'vol_1': list_HAV_test[i]['vol'],
                                'mean_vol_5': list_HAV_test[i]['mean_vol_5'],
                                'mean_vol_22': list_HAV_test[i]['mean_vol_22']})
        predictions.append(HAV_wls_models[i].predict(newdata).tolist())

    df_list = []

    for i in range(len(list_HAV_test)):
        min_length = min(len(list_HAV_test[i]['vol']), len(predictions[i]))
        stock_pre = pd.DataFrame({'time_id': vol[i]['time_id'][502:600],
                                'time_bucket': vol[i]['time_bucket'][502:600],
                                'predicted_volatility': predictions[i][0:min_length]})
        df_list.append(stock_pre)

    combined_df = pd.concat(df_list, ignore_index=True)

    merged_df = pd.merge(stock1, combined_df, on=['time_id', 'time_bucket'], how='left')

    return merged_df


from Prophet.read_parquet import read_parquet

stock1 = read_parquet('output.parquet', [0], "all", False)
stock1.columns = ['time_id', 'time_bucket', 'volatility']
print(dave_HAR_RV(stock1))


