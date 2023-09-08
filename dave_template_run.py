import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from Prophet.read_parquet import read_parquet


def dave_HAR_RV(stock1):

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

    HAV_wls_models = []
    import statsmodels.api as sm

    for i in range(len(vol)):
        model = sm.formula.ols('vol ~ vol_1 + mean_vol_5 + mean_vol_22', data= list_HAV [i]).fit()
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
                                'yhat': predictions[i][0:min_length]})
        df_list.append(stock_pre)

    combined_df = pd.concat(df_list, ignore_index=True)

    forecast_dataframe = pd.merge(stock1, combined_df, on=['time_id', 'time_bucket'], how='left')

    forecast_dataframe['ds'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(pd.timedelta_range(
        start='0 days', periods=len(forecast_dataframe), freq='1S'))
    

    test_data = stock1.copy()
    test_data['volatility'] = test_data.apply(
        lambda row: None if 1 <= row['time_bucket'] <= 502 else row['volatility'], axis=1)
    
    test_data['ds'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(pd.timedelta_range(
        start='0 days', periods=len(test_data), freq='1S'))
    
    train_data = stock1.copy()
    train_data['volatility'] = train_data.apply(
        lambda row: None if 502<= row['time_bucket'] <= 600 else row['volatility'], axis=1)
    
    train_data['ds'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(pd.timedelta_range(
        start='0 days', periods=len(train_data), freq='1S'))


    return forecast_dataframe, test_data, train_data


    #create a time column (it makes matching up all the datasets much easier)
    #data['ds'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(pd.timedelta_range(start='0 days', periods=len(data), freq='1S'))

    #Format Dataset into test train split however you like
    #test_data, train_data = Data_split

    #Train your model
    #model =  object
    
    #Create forecast dataframe
    #forecast_dataframe = pd.DataFrame()


    #Calculate accuracy of your function
    #results = accuracy(forecast_dataframe, test_data)


    # graph(test_data, train_data, forecast_data)


    #return results


def accuracy(forecast: pd.DataFrame, reality: pd.DataFrame) -> dict:

    #This function assumes there are 2 dataframes, one called forecast, and one called reality.
    #reality has columns ds (which is time) and volatility
    #forecast has columns ds (which is time) and yhat, where yhat is the prediction that you're making
    forecast = forecast.rename(columns={'ds': 'time', 'yhat': 'volatility_forecast'})
    reality = reality.rename(columns={'ds' : 'time', 'volatility': 'volatility_reality'})

    forecast = forecast.sort_values(by="time")
    reality = reality.sort_values(by="time")

    #pls dont touch
    merged_data = pd.merge(forecast, reality, on="time")
    merged_data["abs_diff"] = abs(merged_data["volatility_forecast"] - merged_data["volatility_reality"])
    merged_data["diff_squared"] = (merged_data["volatility_forecast"] - merged_data["volatility_reality"])**2
    merged_data["abs_pct_diff"] = abs(merged_data["volatility_forecast"] - merged_data["volatility_reality"]) / merged_data["volatility_reality"]
    merged_data["smape_diff"] = 2 * abs(merged_data["volatility_forecast"] - merged_data["volatility_reality"]) / (abs(merged_data["volatility_forecast"]) + abs(merged_data["volatility_reality"]))

    mae = merged_data["abs_diff"].mean()
    mse = merged_data["diff_squared"].mean()
    rmse = np.sqrt(mse)
    mape = (merged_data["abs_pct_diff"] * 100).mean()
    mdape = np.median(merged_data["abs_pct_diff"] * 100)
    smape = (merged_data["smape_diff"] * 100).mean()

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mdape': mdape, 'smape': smape}



#Graphing is not required, but you can use it for diagnostics
def graph(test_df: pd.DataFrame, train_df: pd.DataFrame, prophet_forecast: pd.DataFrame):
    
    plt.figure(figsize=(15, 7))
    plt.plot(test_df['ds'], test_df['volatility'], label='True Volatility')
    plt.plot(test_df['ds'], prophet_forecast['yhat'], label='Prophet Forecast')
    plt.plot(train_df['ds'], train_df['volatility'], label='Train Volatility')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title('Volatility Forecast')
    plt.legend()
    plt.show()
    pass



#stock1 = read_parquet('output.parquet', [0], "all", False)
#stock1.columns = ['time_id', 'time_bucket', 'volatility']
stock1 = pd.read_csv("/Users/dave/Downloads/stock1_vol.csv")
forecast_dataframe, test_data, train_data= dave_HAR_RV(stock1)
#print(accuracy(forecast_dataframe, test_data))

print(graph(test_data,train_data,forecast_dataframe))












