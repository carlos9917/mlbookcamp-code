# Use this script to send request to flask server,
# either running with flask or with gunicorn
import requests

#url = "http://192.168.0.10:9396/predict"
url = "http://localhost:9696/predict"

customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}
print(requests.post(url, json=customer).json())


