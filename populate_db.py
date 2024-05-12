import yfinance
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from darts import TimeSeries
from darts.models import TiDEModel
from sklearn.model_selection import ParameterGrid
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
import pymongo
import random
warnings.filterwarnings("ignore", module='statsmodels')

uri = os.getenv("MONGO_URI")

client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['arima']

def evaluate_params_tide(params, train_ts, test_ts):
    model = TiDEModel(input_chunk_length=params['input_chunk_length'],
                      output_chunk_length=params['output_chunk_length'],
                      random_state=50,
                      num_encoder_layers=params['num_encoder_layers'],
                      num_decoder_layers=params['num_decoder_layers'],
                      temporal_decoder_hidden=params['temporal_decoder_hidden'],
                      hidden_size=params['hidden_size'],
                      n_epochs=params['n_epochs']
                      )
    model.fit(train_ts)
    prediction = model.predict(len(test_ts))
    # Convert TimeSeries to numpy arrays for calculation
    actual_values = test_ts.values().flatten()
    predicted_values = prediction.values().flatten().tolist()

    # Calculate MSE
    mse = np.mean((actual_values - predicted_values) ** 2)
    return mse, predicted_values

def get_best_params_tide(train_ts, test_ts):
    param_grid = ParameterGrid({
        'input_chunk_length': [10,20,60,120,160],
        'output_chunk_length': [5, 10, 15, 30, 60, 120],
        'num_encoder_layers': [1,2,3],
        'num_decoder_layers': [1,2,3],
        'temporal_decoder_hidden': [32,64],
        'hidden_size': [64, 128, 256],
        'n_epochs': [100, 200]
    })

    for i,params in enumerate(param_grid):
        print('*'*50)
        doc = coll.find_one({"params":params})
        if doc:
            print(f"{str(params)} skipped")
            continue

        results = {}
        results["params"] = params
        results["len_train"] = len(train_ts)
        results["len_test"] = len(test_ts)
        results["test"] = test_ts.values().flatten().tolist()
        results["test_start_date"] = test_ts.start_time().strftime('%Y-%m-%d')
        results["test_end_date"] = test_ts.end_time().strftime('%Y-%m-%d')
        results["trained"] = False

        coll.insert_one(results)

        start = time.time()
        error, prediction = evaluate_params_tide(params, train_ts, test_ts)
        elapsed_time = time.time()-start

        results["elapsed_time"] = elapsed_time
        results["mse"] = error
        results["len_prediction"] = len(prediction)
        results["prediction"] = prediction
        results["trained"] = True

        coll.replace_one({"params":params},results)

        print(f"Iter: {i} took {elapsed_time} with params {params}")


def main_tide():
    df = yfinance.download("^GSPC",start='1990-01-01', end="2024-03-28")
    df['market_value'] = df.Close

    del df['Open'], df['High'], df['Low'], df['Close'], df['Adj Close'], df['Volume']

    series = TimeSeries.from_dataframe(df, time_col=None, value_cols=['market_value'], fill_missing_dates=False, freq='B')
    series_df = series.pd_dataframe()

    # Forward-fill NaN values with the last known value
    series_df.fillna(method='ffill', inplace=True)

    series = TimeSeries.from_dataframe(series_df, time_col=None, value_cols=['market_value'], fill_missing_dates=False, freq='B')

    split_date = pd.Timestamp('2023-01-01')
    train, test = series.split_before(split_date)

    train = train.astype(np.float32)
    test = test.astype(np.float32)

    get_best_params_tide(train, test)


##########################################
def evaluate_params_arima(params, train, test):
    model = ARIMA(train, order=(params['p'],params['q'],params['q']))
    results = model.fit()
    
    start = test.index[0]
    end = test.index[-1]
    prediction = results.predict(start=start, end=end)

    # Calculate MSE
    mse = sum([(y-yp)**2 for y,yp in zip(test,prediction)])/len(test)
    return mse, list(prediction), results


def get_best_params_arima(train_ts, test_ts):
    train = train_ts.pd_series()
    test = test_ts.pd_series()
    param_grid = ParameterGrid({
        'p':[i for i in range(50)],
        'd':[0,1,2,3],
        'q':[i for i in range(50)]
    })
    param_list = list(param_grid)
    random.shuffle(param_list)

    for i,params in enumerate(param_list):
        print('*'*50)
        print(params)
        doc = coll.find_one({"params":params})
        if doc:
            print(f"{str(params)} skipped")
            continue

        results = {}
        results["params"] = params
        results["len_train"] = len(train_ts)
        results["len_test"] = len(test_ts)
        results["test"] = test_ts.values().flatten().tolist()
        results["test_start_date"] = test_ts.start_time().strftime('%Y-%m-%d')
        results["test_end_date"] = test_ts.end_time().strftime('%Y-%m-%d')
        results["trained"] = False

        coll.insert_one(results)

        start = time.time()
        error, prediction, model_fit = evaluate_params_arima(params, train, test)
        elapsed_time = time.time()-start

        results["elapsed_time"] = elapsed_time
        results["mse"] = error
        results["len_prediction"] = len(prediction)
        results["prediction"] = prediction
        results["trained"] = True
        results["model_fit"] = {"aic":str(model_fit.aic), "bic":str(model_fit.bic), "hqic":str(model_fit.hqic), "llf":str(model_fit.llf)}

        coll.replace_one({"params":params},results)

        print(f"Iter: {i} took {elapsed_time} with params {params}")


def main_arima():
    df = yfinance.download("^GSPC",start='1990-01-01', end="2024-03-28")
    df['market_value'] = df.Close

    del df['Open'], df['High'], df['Low'], df['Close'], df['Adj Close'], df['Volume']

    series = TimeSeries.from_dataframe(df, time_col=None, value_cols=['market_value'], fill_missing_dates=False, freq='B')
    series_df = series.pd_dataframe()

    # Forward-fill NaN values with the last known value
    series_df.fillna(method='ffill', inplace=True)

    series = TimeSeries.from_dataframe(series_df, time_col=None, value_cols=['market_value'], fill_missing_dates=False, freq='B')

    split_date = pd.Timestamp('2023-01-01')
    train_ts, test_ts = series.split_before(split_date)

    get_best_params_arima(train_ts, test_ts)


main_arima()