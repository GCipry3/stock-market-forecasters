import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
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

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-28', split_date='2023-01-01'):
    df = yf.download(ticker, start=start_date, end=end_date)['Close']
    df = df.fillna(method='ffill')
    df = (df - df.mean()) / df.std()
    train = df[:split_date]
    test = df[split_date:]
    return train, test

def train_model(train_series, lr, epochs):
    X_train = torch.tensor(train_series.index.factorize()[0], dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(train_series.values, dtype=torch.float32).view(-1, 1)
    
    model = LinearRegressionModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model

def evaluate_model(model, test_series):
    X_test = torch.tensor(test_series.index.factorize()[0], dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(test_series.values, dtype=torch.float32).view(-1, 1)
    
    model.eval()
    predictions = model(X_test).detach().numpy()
    error = mean_squared_error(y_test, predictions)
    
    return error, predictions

def log(message):
    with open("linear_regression_logs.txt", "a") as f:
        f.write(f"{message}\n")

def start_grid_search():
    param_grid = ParameterGrid({
        'lr': [0.01, 0.001, 0.0001],
        'epochs': [50, 100, 200]
    })
    param_list = list(param_grid)
    random.shuffle(param_list)

    total_params = len(param_list)
    log(f"Starting grid search over {total_params} combinations.")

    for i, params in enumerate(param_list):
        log(f"\n{'*'*50}\nEvaluating Parameters: {params} ({i+1}/{total_params})")
        query = {"params": params, "model": "LinearRegression"}
        doc = coll.find_one(query)
        
        if doc and doc.get("trained"):
            log(f"Parameters {params} already evaluated and trained. Skipping.")
            continue
        elif doc:
            log(f"Found incomplete record for {params}. Deleting and re-evaluating.")
            coll.delete_one(query)

        train_series, test_series = load_data()
        results = {
            "model": "LinearRegression",
            "params": params,
            "len_train": len(train_series),
            "len_test": len(test_series),
            "test": test_series.tolist(),
            "test_start_date": test_series.index[0].strftime('%Y-%m-%d'),
            "test_end_date": test_series.index[-1].strftime('%Y-%m-%d'),
            "trained": False
        }
        coll.insert_one(results)

        try:
            start_time = time.time()
            log("Training Linear Regression model...")
            model = train_model(train_series, lr=params["lr"], epochs=params["epochs"])
            error, predictions = evaluate_model(model, test_series)
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
            log(f"Error during training or prediction with params {params}: {e}")
            results["error"] = str(e)
            coll.replace_one({"params": params}, results)

if __name__ == '__main__':
    start_grid_search()
