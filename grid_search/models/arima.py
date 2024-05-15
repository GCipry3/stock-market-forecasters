import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from darts import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from darts.metrics import mse
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import random
from dotenv import load_dotenv
import os
import pymongo
import time

load_dotenv()
uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['SP500_forecast_models']

warnings.filterwarnings("ignore", module='statsmodels')

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-20', split_date='2020-01-01', apply_filter=False, debug=False):
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
    """ Applies a zero-phase Butterworth filter to a given pandas Series. """
    b, a = butter(order, critical_frequency, btype='low', analog=False)
    filtered_values = filtfilt(b, a, series)
    filtered_series = pd.Series(filtered_values, index=series.index)
    if debug:
        print("✅ Zero-phase Butterworth filter applied.")
    return filtered_series

def train_arima(train_series, order=(1, 1, 1)):
    """ Train an ARIMA model on the provided training data. """
    model = ARIMA(train_series, order=order)
    results = model.fit()
    return results

def generate_predictions(model, start, end):
    """ Generate predictions using the trained ARIMA model. """
    predictions = model.predict(start=start, end=end, typ='levels')
    return predictions

def plot_predictions(test_series, prediction, title=""):
    plt.figure(figsize=(12, 6))
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_series(train_series, test_series, prediction, title=""):
    """ Plot the training data, test data, and ARIMA predictions. """
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label='Train')
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()


def log(message):
    with open("arima_logs.txt","a") as f:
        f.write(f"{message}\n")


def start_grid_search():
    param_grid = ParameterGrid({
        'p':range(25),
        'd':range(4),
        'q':range(25),
        'filter':[True,False]
    })
    param_list = list(param_grid)
    random.shuffle(param_list)

    total_params = len(param_list)
    log(f"Starting grid search over {total_params} combinations.")

    for i, params in enumerate(param_list):
        log(f"\n{'*'*50}\nEvaluating Parameters: {params} ({i+1}/{total_params})")
        query = {"params": params, "model": "ARIMA"}
        doc = coll.find_one(query)
        
        if doc and doc.get("trained"):
            log(f"Parameters {params} already evaluated and trained. Skipping.")
            continue
        elif doc:
            log(f"Found incomplete record for {params}. Deleting and re-evaluating.")
            coll.delete_one(query)

        train_series, test_series = load_data(apply_filter=params["filter"], debug=True)

        results = {
            "model": "ARIMA",
            "params": params,
            "len_train": len(train_series),
            "len_test": len(test_series),
            "test": test_series.tolist(),
            "test_start_date": train_series.index[0].strftime('%Y-%m-%d'),
            "test_end_date": test_series.index[-1].strftime('%Y-%m-%d'),
            "trained": False
        }
        coll.insert_one(results)

        try:
            start = time.time()
            log("Training ARIMA model...")
            model = train_arima(train_series, order=(params["p"], params["d"], params["q"]))
            predictions = generate_predictions(model, start=test_series.index[0], end=test_series.index[-1])
            error = mean_squared_error(test_series, predictions)
            elapsed_time = time.time() - start

            results.update({
                "elapsed_time": elapsed_time,
                "mse": error,
                "len_prediction": len(predictions),
                "prediction": predictions.tolist(),
                "trained": True
            })
            coll.replace_one({"params": params}, results)
            log(f"Completed: {i+1}/{total_params}. Elapsed time: {elapsed_time:.2f}s. MSE: {error:.4f}")
        except Exception as e:
            log(f"Error during training or prediction with params {params}: {e}")
            results["error"] = str(e)
            coll.replace_one({"params": params}, results)

if __name__ == '__main__':
    start_grid_search()