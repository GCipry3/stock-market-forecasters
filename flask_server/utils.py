import yfinance as yf
import pandas as pd
from darts import TimeSeries
import numpy as np
from scipy.signal import butter, filtfilt

def load_data(ticker="^GSPC", start_date='1990-01-02', end_date='2024-03-20', split_date='2023-01-01', apply_filter=False, debug=False):
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

