import pickle

#import pandas as pd
#import numpy as np
#
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import roc_auc_score


model_file = 'model1.bin'
dv_file = 'dv.bin'
with open(model_file, 'rb') as f_in:
     model = pickle.load(f_in)
with open(dv_file, 'rb') as f_in:
     dv = pickle.load(f_in)

def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    #churn = y_pred >= 0.5

    #result = {
    #    'churn_probability': float(y_pred),
    #    'churn': bool(churn)
    #}

    return y_pred # jsonify(result)


customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
print(predict(customer))

