#Homework 2, part 2
import numpy as np
import pandas as pd
import reg_utils as rutils
data = pd.read_csv("AB_NYC_2019.csv")

# Data splitting
log_price = np.log1p(data["price"])

"""
Question 5
We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
For each seed, do the train/validation/test split with 60%/20%/20% distribution.
Fill the missing values with 0 and train a model without regularization.
For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
Round the result to 3 decimal digits (round(std, 3))
"""
col = "reviews_per_month" #This is  one of the cols with 10052 values missing
df = data.copy()
base=['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

df[col] = df[col].fillna(0)
all_scores = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    df_train,df_val,df_test = rutils.split_data(df,seed=seed)
    #Apply the log transformation to the price variable using the np.log1p() function.
    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)
    
    #select as base all columns except price
    #drop the price column before doing anything
    del df_train['price']
    del df_val['price']
    del df_test['price']
    
    X_train = rutils.prepare_X(df_train,base)
    X_val = rutils.prepare_X(df_val,base)
    w_0, w = rutils.train_linear_regression(X_train, y_train)
    y_pred = w_0 + X_train.dot(w)
    y_pred = w_0 + X_val.dot(w)
    rmse = round(rutils.rmse(y_val, y_pred),2)
    all_scores.append(rmse)
    print(f'RMSE (no reg) with seed {seed}: {rmse}')

print(f"stdev of all scores: {np.std(all_scores)}")

"""
Question 6
Split the dataset like previously, use seed 9.
Combine train and validation datasets.
Train a model with r=0.001.
What's the RMSE on the test dataset?
"""
df_train,df_val,df_test = rutils.split_data(df,seed=9)
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)
del df_train['price']
del df_val['price']
del df_test['price']
X_train = rutils.prepare_X(df_train,base)
w_0, w = rutils.train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
rmse = round(rutils.rmse(y_train, y_pred),2)
print(f'RMSE (no reg) with seed 9: {rmse}')
