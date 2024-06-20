from dotenv import load_dotenv
import os
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()
uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(uri)
db = client["Licenta"]
coll = db['SP500_forecast_models']

doc_mse=18782.96704664109
unfiltered_test = coll.find_one({"mse":doc_mse})['test']

cursor = coll.find({})
counter = 0

for document in cursor:
    counter+=1
    id = document["_id"]
    coll.update_one({"_id": id}, {"$set":{"unfiltered_test":unfiltered_test}})
    if counter%10==0:
        print(counter)