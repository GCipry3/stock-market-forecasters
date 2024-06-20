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

cursor = coll.find()
counter = 0
for document in cursor:
    counter+=1
    id = document["_id"]
    
    y = np.array(document['unfiltered_test'])
    yp = np.array(document['prediction'])
    
    r_matrix = np.corrcoef(y, yp)
    r = r_matrix[0, 1]
    
    coll.update_one({"_id": id}, {"$set": {"r": r}})
    if counter%10==0:
        print(counter)
