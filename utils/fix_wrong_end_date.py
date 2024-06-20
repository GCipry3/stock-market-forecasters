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

nr_of_days = 324
last_day = "2024-04-02"

cursor = coll.find({"len_test":{"$ne":nr_of_days}})
counter = 0

def log(message):
    with open('log.txt','a') as f:
        f.write(f'{message}\n')
#660c5618ae9d8111f91af62f
#3821.396484375
for document in cursor:
    counter+=1
    id = document["_id"]

    document['prediction']=document['prediction'][:nr_of_days]
    document['test']=document['test'][:nr_of_days]
    document['test_end_date']=last_day
    document['len_test']=nr_of_days
    document['len_prediction']=nr_of_days

    coll.replace_one({"_id": id}, document)
    log(id)

    if counter%10==0:
        print(counter)