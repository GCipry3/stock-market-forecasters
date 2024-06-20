from dotenv import load_dotenv
import os
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import butter, filtfilt

load_dotenv()
uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['SP500_forecast_models']

def plot_predictions(test_series, prediction, title=""):
    plt.figure(figsize=(12, 6))
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_series(train_series, test_series, prediction, title=""):
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label='Train')
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-28', split_date='2023-01-01', apply_filter=False, debug=False):
    """ Load data, perform forward fill, optionally apply a Butterworth filter, and split into train and test. """
    # Download historical data from Yahoo Finance
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
        print("✅ Zero-phase Butterworth filter applied.")
    return filtered_series

def display_doc(mse):
    doc = coll.find_one({"mse":mse})

    y = np.array(doc['unfiltered_test'])
    yp = np.array(doc['prediction'])
    r = np.corrcoef(y, yp)[0, 1]

    all_business_days = pd.date_range(start=doc['test_start_date'], end=doc['test_end_date'], freq='B')

    y_series = pd.Series(y,index=all_business_days)
    yp_series = pd.Series(yp,index=all_business_days)
    model = doc['model']
    filter = doc['params']['filter']
    title = f'{model} model' if not filter else f'{model} model with Butterworth Filter'
    plot_predictions(y_series, yp_series, title=title)


def display_full_doc(mse):
    doc = coll.find_one({"mse":mse})

    y = np.array(doc['unfiltered_test'])
    yp = np.array(doc['prediction'])

    all_business_days = pd.date_range(start=doc['test_start_date'], end=doc['test_end_date'], freq='B')

    y_series = pd.Series(y,index=all_business_days)
    yp_series = pd.Series(yp,index=all_business_days)
    train_series , _ = load_data(apply_filter=doc['params']['filter'])
    model = doc['model']
    filter = doc['params']['filter']
    title = f'{model} model' if not filter else f'{model} model with Butterworth Filter'
    plot_full_series(train_series, y_series, yp_series, title=title)

#display_doc(18782.96704664109)

display_full_doc(1940992.2951213908)