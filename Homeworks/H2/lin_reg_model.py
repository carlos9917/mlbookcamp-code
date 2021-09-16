#Homework 2
import numpy as np
import pandas as pd
import reg_utils as rutils
data = pd.read_csv("AB_NYC_2019.csv")

# look for missing values
print(f"Number of lines in data set: {data.shape[0]}")
# Q1. Look for column with missing values
# I get two columns with 10000+ missing values
#last_review is missing 10052 missing values
#reviews_per_month is missing 10052 missing values

for col in data.columns:
    missing = [notfound for notfound in data[col].isnull() if notfound]
    if len(missing) != 0: 
        print(f"{col} is missing {len(missing)} missing values")
        #if len(missing) <= 20: #only print if it fits on screen!
        #    check = data.mask(data[col].isnull(), 9999)[col]
        #    indexes = check[check==9999].index.values
        #    mdata = [data.iloc[i] for i in indexes]
        #    print(f"Missing data: {mdata}")

# Question 2
#What's the median (50% percentile) for variable 'minimum_nights'?
med = data["minimum_nights"].median()
print(f"median for minimum nights: {med}")

# Data splitting
log_price = np.log1p(data["price"])

# Drop the price column
#data.drop(columns=["price"],inplace=True)

# Q3. Deal with missing values for the column from Q1.
# Try the two options: fill it with 0 or with the mean of this variable.
col = "reviews_per_month" #This is  one of the cols with 10052 values missing
#Use 0 
#df= data[]
df = data.copy()
base=['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

mean_col = df[col].mean()
df[col] = df[col].fillna(0)
df_train,df_val,df_test = rutils.split_data(df)
#Apply the log transformation to the price variable using the np.log1p() function.
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

#select as base all columns except price
#drop the price column before doing anything
del df_train['price']
del df_val['price']
del df_test['price']

#base=df_train.columns

X_train = rutils.prepare_X(df_train,base)
#df_train.drop(columns=["price"],inplace=True)
#df_val.drop(columns=["price"],inplace=True)
#df_test.drop(columns=["price"],inplace=True)
w_0, w = rutils.train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
print('validation:', round(rutils.rmse(y_train, y_pred),2))
rutils.plot_compare(y_train,y_pred,"lin1_0fill.png")

# now using regularizationa
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = rutils.train_linear_regression_reg(X_train, y_train,r)
    y_pred = w_0 + X_train.dot(w)
    print(f'validation using r={r}: {round(rutils.rmse(y_train, y_pred),2)}')



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
