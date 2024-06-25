import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import butter, filtfilt

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-28', split_date='2023-01-01', apply_filter=False, debug=False):
    df = yf.download(ticker, start=start_date, end=end_date)['Close']
    
    # Fill missing values and ensure all business days are accounted for
    all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    df = df.reindex(all_business_days).ffill()  # Forward filling the missing values
    if apply_filter:
        df = apply_butterworth_filter(df, debug=debug)

    if debug:
        print(f"Data loaded and forward filled. Total rows after forward fill: {len(df)}.")

    # Split the data into train and test sets
    split_point = pd.Timestamp(split_date)
    train_series = df[df.index < split_point]
    test_series = df[df.index >= split_point]
    return train_series, test_series

def apply_butterworth_filter(series, order=3, critical_frequency=0.05, debug=False):
    """ Applies Butterworth filter to a given pandas Series. """
    b, a = butter(order, critical_frequency, btype='low', analog=False)
    filtered_values = filtfilt(b, a, series)
    filtered_series = pd.Series(filtered_values, index=series.index)
    if debug:
        print(" Butterworth filter applied.")
    return filtered_series

def train_model(train_series, use_gradient=False):
    if use_gradient:
        return gradient_descent(train_series)
    else:
        return linear_regression_without_gradient(train_series)


def get_mse(y_pred, y_real):
    return np.mean((y_real - y_pred) ** 2)

def linear_regression_without_gradient(train_series):
    N = len(train_series)
    x = pd.Series(range(N))
    y = train_series.reset_index(drop=True)
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    b1 = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    b0 = y_mean - b1 * x_mean
    
    return b0, b1



################## NEEDS FIX
def get_mse_deriv(x, y, b0, b1):
    return np.mean((y-(b0+b1*x))**2)

def aprox_derivative(x, y, b0, b1, func=get_mse_deriv, derivate = 'b0'):
    delta = 0.1
    if derivate == 'b0':
        return (func(x,y,b0+delta,b1) - func(x,y,b0,b1))/delta
    elif derivate == 'b1':
        return (func(x,y,b0,b1+delta) - func(x,y,b0,b1))/delta

    raise Exception(f"Unkown derivate coef: {derivate}")
def gradient_descent(train_series, lr=1e-5, iterations=1000):
    N = len(train_series)
    x = pd.Series(range(N))
    y = train_series.reset_index(drop=True)
    
    b0 = np.random.randn()
    b1 = np.random.randn()
    
    prev_error = None
    for epoch in range(iterations):
        y_pred = b0 + b1 * x
        error = get_mse(y_pred, y)
        b0_grad = -2 * (y - y_pred).sum() / N
        b1_grad = -2 * (x * (y - y_pred)).sum() / N
        b0 -= lr * b0_grad
        b1 -= lr * b1_grad
        
        if epoch % 100 == 0 or epoch == iterations - 1:
            print(f"Epoch {epoch}:\n\tMSE={error}\n\tb0={b0}\n\tb1={b1}")
            print(f"\tb0_grad={b0_grad}\n\tb1_grad={b1_grad}")

        if prev_error and abs(prev_error - error) < 1e-6:
            print(f"Model converged at epoch {epoch}.")
            break
        
        prev_error = error

    print('-' * 50)
    return b0, b1
################## NEEDS FIX

def generate_predictions(train_series, test_series, start=None, end=None, debug=False):
    b0, b1 = train_model(train_series=train_series, use_gradient=False)
    start_date = start if start else test_series.index[0]
    end_date = end if end else test_series.index[-1]
    all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    if debug:
        print(f"Generating predictions between {start_date} and {end_date}\n\tb0 = {b0}\n\tb1 = {b1}")
    x = pd.Series(range(len(all_business_days)), index=all_business_days)
    predictions = b0 + b1 * x
    if debug:
        print(predictions.head())
        print(f" Forecast generated.")
    return predictions


def plot_predictions(test_series, prediction, title=""):
    prediction = prediction[prediction.index >= test_series.index[0]]
    plt.figure(figsize=(12, 6))
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_series(train_series, test_series, prediction, title=""):
    """ Plot the training data, test data, and Linear Regression predictions. """
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label='Train')
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()


train_series, test_series = load_data(apply_filter=True, debug=True)
print(len(test_series))
predictions = generate_predictions(train_series, test_series, debug=True, start=train_series.index[0], end=test_series.index[-1])
error = get_mse(predictions, test_series)

print(f"MSE: {error}")
#plot_predictions(test_series, predictions, title="Linear Regression model with Butterworth Filter")
plot_full_series(train_series, test_series, predictions, title="Linear Regression Model Forecast with Butterworth Filter")