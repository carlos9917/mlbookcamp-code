# Use this script to send request to flask server,
# either running with flask or with gunicorn
# Run this to answer Q6 when server is up
import requests
url = "http://localhost:9696/predict"
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}
print(requests.post(url, json=customer).json())
requests.post(url, json=customer).json()


