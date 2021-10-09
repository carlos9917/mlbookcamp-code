import pickle

#Load the model
model_file = 'model1.bin'
dv_file = 'dv.bin'
with open(model_file, 'rb') as f_in:
     model = pickle.load(f_in)
with open(dv_file, 'rb') as f_in:
     dv = pickle.load(f_in)


#create Flask app
from flask import Flask
from flask import request
from flask import jsonify

#import requests
app = Flask('churn')
@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__=="__main__":
    #url = 'http://localhost/predict'
    app.run(debug=True, host='0.0.0.0', port=9696)

    #Then I open this like: http://192.168.0.10:9696/predict
    #But I will get an error
    #Need to get the request from another script: python send_request.py


