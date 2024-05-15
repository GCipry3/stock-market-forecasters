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

class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(PolynomialRegressionModel, self).__init__()
        self.poly = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.poly(x)

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-28', split_date='2023-01-01', degree=2):
    df = yf.download(ticker, start=start_date, end=end_date)['Close']
    df = df.fillna(method='ffill')
    df = (df - df.mean()) / df.std()
    train = df[:split_date]
    test = df[split_date:]
    return create_polynomial_features(train, degree), create_polynomial_features(test, degree)

def create_polynomial_features(series, degree):
    indices = np.arange(len(series))
    poly_features = np.array([indices**i for i in range(1, degree+1)]).T
    poly_df = pd.DataFrame(poly_features, index=series.index)
    poly_df['Close'] = series.values
    return poly_df

def train_model(train_df, lr, epochs):
    X_train = torch.tensor(train_df.drop('Close', axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Close'].values, dtype=torch.float32).view(-1, 1)
    
    model = PolynomialRegressionModel(X_train.shape[1])
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

def evaluate_model(model, test_df):
    X_test = torch.tensor(test_df.drop('Close', axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Close'].values, dtype=torch.float32).view(-1, 1)
    
    model.eval()
    predictions = model(X_test).detach().numpy()
    error = mean_squared_error(y_test, predictions)
    
    return error, predictions

def log(message):
    with open("polynomial_regression_logs.txt", "a") as f:
        f.write(f"{message}\n")

def start_grid_search():
    param_grid = ParameterGrid({
        'lr': [0.01, 0.001, 0.0001],
        'epochs': [50, 100, 200],
        'degree': [2, 3, 4]
    })
    param_list = list(param_grid)
    random.shuffle(param_list)

    total_params = len(param_list)
    log(f"Starting grid search over {total_params} combinations.")

    for i, params in enumerate(param_list):
        log(f"\n{'*'*50}\nEvaluating Parameters: {params} ({i+1}/{total_params})")
        query = {"params": params, "model": "PolynomialRegression"}
        doc = coll.find_one(query)
        
        if doc and doc.get("trained"):
            log(f"Parameters {params} already evaluated and trained. Skipping.")
            continue
        elif doc:
            log(f"Found incomplete record for {params}. Deleting and re-evaluating.")
            coll.delete_one(query)

        train_df, test_df = load_data(degree=params["degree"])
        results = {
            "model": "PolynomialRegression",
            "params": params,
            "len_train": len(train_df),
            "len_test": len(test_df),
            "test": test_df['Close'].tolist(),
            "test_start_date": test_df.index[0].strftime('%Y-%m-%d'),
            "test_end_date": test_df.index[-1].strftime('%Y-%m-%d'),
            "trained": False
        }
        coll.insert_one(results)

        try:
            start_time = time.time()
            log("Training Polynomial Regression model...")
            model = train_model(train_df, lr=params["lr"], epochs=params["epochs"])
            error, predictions = evaluate_model(model, test_df)
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
