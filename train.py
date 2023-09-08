from vol import vol
from pmdarima.model_selection import train_test_split
import pmdarima as pm

#takes in data and returns 

def train(stock1, training_ratio: float):

    y = vol(stock1)
    #split data
    train, test = train_test_split(y, train_size=float(training_ratio))

    # Fit the model automatically
    model = pm.auto_arima(train, maxiter=100)



    return model
