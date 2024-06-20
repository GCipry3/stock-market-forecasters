from flask import Flask, render_template, request, redirect, url_for
import pymongo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from darts import TimeSeries
import io
import base64
from bson.objectid import ObjectId

matplotlib.use('Agg')

load_dotenv()

app = Flask(__name__)
uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['SP500_forecast_models']

def plot_forecast_only(test_ts: TimeSeries, prediction: TimeSeries, title):
    plt.figure(figsize=(10, 6))
    test_ts.plot(label='Test', lw=0.5)
    prediction.plot(label='Predictions', lw=0.5)
    
    actual_values = test_ts.values().flatten()
    predicted_values = prediction.values().flatten()
    error = np.mean((actual_values - predicted_values) ** 2)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Forecast with MSE: {error:.2f}')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def extract_and_plot(doc):
    test_array = doc['test']
    prediction_array = doc['prediction']
    start_date = doc['test_start_date']
    end_date = doc['test_end_date']
    model = doc['model']
    params = doc['params']
    filter = params['filter']
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    test_ts = TimeSeries.from_times_and_values(business_days, np.array(test_array))
    prediction_ts = TimeSeries.from_times_and_values(business_days, np.array(prediction_array))

    title = f'{model} model' if not filter else f'{model} model with Butterworth Filter'
    return plot_forecast_only(test_ts, prediction_ts, title=title), doc['params']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/manual_check/<model>')
def manual_check(model):
    query = {"model": model, "mse": {"$exists": True}, "manual_status": {"$exists": False}}
    document = coll.find_one(query)
    if document:
        plot_url, params = extract_and_plot(document)
        processed_docs = coll.count_documents({"model": model, "manual_status": {"$exists": True}})
        accepted = coll.count_documents({'model': model, 'manual_status': 'Accepted'})
        maybe = coll.count_documents({'model': model, 'manual_status': 'Maybe'})
        rejected = coll.count_documents({'model': model, 'manual_status': 'Rejected'})
        total_docs = coll.count_documents(query)
        return render_template('index.html', plot_url=plot_url, params=params, doc_id=document['_id'],
                               total_docs=total_docs, processed_docs=processed_docs, accepted=accepted, maybe=maybe, rejected=rejected)
    else:
        return f'No more {model} documents to process!<br><a href="/">HOME PAGE</a><br>'

@app.route('/display/<model>/<status>')
def display_models(model, status):
    limit = request.args.get('limit',10)

    query = {"model": model, "mse": {"$exists": True}, "manual_status": status}
    documents = coll.find(query).limit(int(limit))
    plots_params = []
    for document in documents:
        plot_url, params = extract_and_plot(document)
        metrics = {'mse':document['mse'], 'r':document['r']}
        plots_params.append((plot_url, params, metrics))
    return render_template('accepted.html', plots_params=plots_params)

@app.route('/update_status/<doc_id>/<status>')
def update_status(doc_id, status):
    document = coll.find_one({"_id": ObjectId(doc_id)})
    if document:
        document['manual_status'] = status
        coll.update_one({"_id": ObjectId(doc_id)}, {"$set": {"status_added": True, "manual_status": status}})
    return redirect(url_for('manual_check', model=document['model']))

@app.route('/tide/docs')
def tide_docs():
    return render_template('tide_docs.html')

if __name__ == '__main__':
    app.run(debug=True)
