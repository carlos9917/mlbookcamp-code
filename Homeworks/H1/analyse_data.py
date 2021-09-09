#Homework 1
import numpy as np
import pandas as pd
print(f"Current version of pandas {pd.__version__}")
print(f"Current version of numpy {np.__version__}")
data = pd.read_csv("data.csv")
bmw = data[data.Make == "BMW"]["MSRP"].mean()
print(f"Average price of the BMW: {bmw}")
data2015 = data[data.Year >= 2015]
missing = data2015["Engine HP"].shape[0] - data2015["Engine HP"].dropna().shape[0]
print(f"Missing values {missing}")
meanHP = data2015["Engine HP"].mean()
meanHP_after = data["Engine HP"].fillna(value=meanHP).mean()
print(f"Mean before filling: {meanHP}")
print(f"Mean after filling: {meanHP_after}")
rolls = data[data["Make"] == "Rolls-Royce"]
X = rolls[["Engine HP", "Engine Cylinders", "highway MPG"]].drop_duplicates().to_numpy()
XTX = np.matmul(X.T,X)
XTXinv = np.linalg.inv(XTX)
print(f"Sum of values of the inverse: {XTXinv.sum()}")

#Create an array y with values [1000, 1100, 900, 1200, 1000, 850, 1300].
#Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
#What's the value of the first element of w?.
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
M = np.matmul(XTXinv,X.T)
w = np.matmul(M,y) 
print(f"First value of w: {w[0]}")
