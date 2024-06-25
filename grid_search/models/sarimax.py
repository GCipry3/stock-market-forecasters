import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import warnings
import random
from dotenv import load_dotenv
import os
import pymongo
import time

# Load environment variables
load_dotenv()
uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['SP500_forecast_models']

warnings.filterwarnings("ignore", module='statsmodels')

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-28', split_date='2023-01-01', apply_filter=False, debug=False):
    """ Load data, perform forward fill, optionally apply a Butterworth filter, and split into train and test. """
    df = yf.download(ticker, start=start_date, end=end_date)['Close']
    
    all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    df = df.reindex(all_business_days).ffill()  # Forward filling the missing values
    if apply_filter:
        df = apply_butterworth_filter(df, debug=debug)

    if debug:
        print(f"Data loaded and forward filled. Total rows after forward fill: {len(df)}.")

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

def train_sarimax(train_series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    """ Train a SARIMAX model on the provided training data. """
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results

def generate_predictions(model, start, end):
    """ Generate predictions using the trained SARIMAX model. """
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
    """ Plot the training data, test data, and SARIMAX predictions. """
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label='Train')
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(prediction.index, prediction, label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()

def log(message):
    with open("sarimax_logs.txt", "a") as f:
        f.write(f"{message}\n")

def start_grid_search():
    param_grid = ParameterGrid({
        'p': range(0, 6),
        'd': range(0, 2),
        'q': range(0, 6),
        'P': range(0, 3),
        'D': range(0, 2),
        'Q': range(0, 3),
        's': [5, 10, 20, 30],
        'filter': [True, False]
    })
    param_list = list(param_grid)
    random.shuffle(param_list)

    total_params = len(param_list)
    log(f"Starting grid search over {total_params} combinations.")

    for i, params in enumerate(param_list):
        log(f"\n{'*'*50}\nEvaluating Parameters: {params} ({i+1}/{total_params})")
        query = {"params": params, "model": "SARIMAX"}
        doc = coll.find_one(query)
        
        if doc and doc.get("trained"):
            log(f"Parameters {params} already evaluated and trained. Skipping.")
            continue
        elif doc:
            log(f"Found incomplete record for {params}. Deleting and re-evaluating.")
            coll.delete_one(query)

        try:
            train_series, test_series = load_data(apply_filter=params["filter"], debug=True)

            results = {
                "model": "SARIMAX",
                "params": params,
                "len_train": len(train_series),
                "len_test": len(test_series),
                "test": test_series.tolist(),
                "test_start_date": test_series.index[0].strftime('%Y-%m-%d'),
                "test_end_date": test_series.index[-1].strftime('%Y-%m-%d'),
                "trained": False
            }
            coll.insert_one(results)

            start_time = time.time()
            log("Training SARIMAX model...")
            model = train_sarimax(train_series, order=(params["p"], params["d"], params["q"]),
                                  seasonal_order=(params["P"], params["D"], params["Q"], params["s"]))
            predictions = generate_predictions(model, start=test_series.index[0], end=test_series.index[-1])
            error = mean_squared_error(test_series, predictions)
            elapsed_time = time.time() - start_time

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
            results["error"] = str(e)
            coll.replace_one({"params": params}, results)

if __name__ == '__main__':
    start_grid_search()
