import yfinance as yf
import pandas as pd
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mse
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import ParameterGrid
import random
import os
import pymongo
import time
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['SP500_forecast_models']

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj


def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-20', split_date='2020-01-01', apply_filter=False, debug=False):
    df = yf.download(ticker, start=start_date, end=end_date)['Close']
    df = df.reindex(pd.date_range(start=start_date, end=end_date, freq='B')).ffill()
    if apply_filter:
        df = apply_butterworth_filter(df)
    split_point = pd.Timestamp(split_date)
    train_series = df[:split_point]
    test_series = df[split_point:]
    return TimeSeries.from_series(train_series).astype(np.float32), TimeSeries.from_series(test_series).astype(np.float32)

def apply_butterworth_filter(series, order=3, critical_frequency=0.05, debug=False):
    b, a = butter(order, critical_frequency, btype='low', analog=False)
    filtered_values = filtfilt(b, a, series)
    return pd.Series(filtered_values, index=series.index)

def train_tide(train_ts, params):
    model = TiDEModel(**params)
    model.fit(train_ts)
    return model

def evaluate_model(model, test_ts):
    prediction = model.predict(len(test_ts))
    error = mse(test_ts, prediction)
    return prediction, error


def log(message):
    with open("tide_logs.txt","a") as f:
        f.write(f"{message}\n")


def start_grid_search():
    param_grid = ParameterGrid({
        'hidden_size': [64, 128, 256],
        'input_chunk_length': [30, 60, 90],
        'output_chunk_length': [10, 20, 30],
        'n_epochs': [100,300,500,800],
        'num_decoder_layers': [1, 2, 3],
        'num_encoder_layers': [1, 2, 3],
        'temporal_decoder_hidden': [32, 64],
        'random_state': [50],
        'show_warnings': [True],
        'filter': [True, False]
    })
    param_list = list(param_grid)
    random.shuffle(param_list)

    total_params = len(param_list)
    log(f"Starting grid search over {total_params} combinations.")
    
    for params in param_list:
        log(f"Evaluating parameters: {params}")
        query = {"params": params, "model": "TiDE"}
        doc = coll.find_one(query)
        if doc and doc.get("trained"):
            log(f"Parameters {params} already evaluated and trained. Skipping.")
            continue
        elif doc:
            log(f"Found incomplete record for {params}. Deleting and re-evaluating.")
            coll.delete_one(query)
        
        params_copy = params.copy()
        train_ts, test_ts = load_data(apply_filter=params_copy.pop('filter'))

        results = {
            "model": "TiDE",
            "params": params,
            "len_train": len(train_ts),
            "len_test": len(test_ts),
            "test": test_ts.values().flatten().tolist(),
            "test_start_date": test_ts.start_time().strftime('%Y-%m-%d'),
            "test_end_date": test_ts.end_time().strftime('%Y-%m-%d'),
            "trained": False
        }
        coll.insert_one(results)

        try:
            start = time.time()
            log("Training TiDE model...")
            model = train_tide(train_ts, params_copy)
            prediction, error = evaluate_model(model, test_ts)
            elapsed_time = time.time()-start
            results.update({
                "elapsed_time": elapsed_time,
                "params": convert_numpy(params),
                "mse": convert_numpy(error),
                "prediction": convert_numpy(prediction.values().flatten()),
                "trained": True
            })

            coll.replace_one({"params": params}, results)
            log(f"Parameters {params} trained with MSE: {error}")
        except Exception as e:
            log(f"Error during training or prediction with params {params}: {e}")
            results["error"] = str(e)
            coll.replace_one({"params": params}, results)

if __name__ == '__main__':
    start_grid_search()
