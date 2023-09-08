import numpy as np
from vol import vol
import matplotlib.pyplot as plt
from pmdarima.model_selection import train_test_split




def forecast(model, data, training_ratio):

    y = vol(data)
    training_num = round(training_ratio*len(y))
    train, test = train_test_split(y, train_size=float(training_ratio))

    # make the forecasts
    forecast = model.predict(test.shape[0])  # predict N steps into the future

    plt.figure(figsize  = (8,8))
    plt.title('Forecast result and Test data') 

    # Visualize the forecasts (blue=train, green=forecasts)
    x = np.arange(y.shape[0])
    plt.plot(x[:training_num], train, c='blue')
    plt.plot(x[training_num:], forecast, c='green', label = 'ARIMA Forecast')
    plt.plot(x[training_num:], test, c='red', label = 'Test Data')
    plt.legend()
    plt.show()

    return forecast