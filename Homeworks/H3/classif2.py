#Homework 3. Questions 1-6
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import H2.reg_utils as rutils
data = pd.read_csv("../H2/AB_NYC_2019.csv")
base=['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

#Q1: What is the most frequent observation (mode) for the column 'neighbourhood_group'?
sel_col="neighbourhood_group"
most_common = data[sel_col].mode()[0]
print(f"Most common observation in {sel_col}: {most_common}")
#make a copy of dataframe and replace missing values
df = data.copy()
df.fillna(0,inplace=True)

"""
Split the data
Split your data in train/val/test sets, with 60%/20%/20% distribution.
Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
Make sure that the target value ('price') is not in your dataframe.
"""
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

# Make price binary
# create variable above_average, which is 1 if price above 152
#Doing this before dropping price  below
df_train['above_average'] = np.where(df_train['price'] < 150, 1, 0)
df_val['above_average'] = np.where(df_val['price'] < 150, 1, 0)
df_test['above_average'] = np.where(df_test['price'] < 150, 1, 0)

categorical_columns = list(df[base].dtypes[df[base].dtypes == 'object'].index)
numerical_columns = list(df[base].dtypes[df[base].dtypes == 'float'].index) + list(df[base].dtypes[df[base].dtypes == 'int'].index)
numerical_columns.remove("price")
print(f"{len(categorical_columns)} categorical columns: {categorical_columns}")
print(f"{len(numerical_columns)} numerical columns: {numerical_columns}")


#Q2 Variables with biggest corr
corrs=df_full_train[categorical_columns + numerical_columns].corrwith(df_full_train.price).to_frame('correlation')
print("Looking for variables with biggest correlation")
print(corrs)

#

# Q3 Calculate the mutual information score with the (binarized) price for the two categorical variables that we have. Use the training set only.
from sklearn.metrics import mutual_info_score
#Apply the mutual info score to price
#def mutual_info_price_score(series):
#    return mutual_info_score(series, df_full_train.above_average)
#mi = df_full_train[categorical_columns].apply(mutual_info_price_score)
#bgg=mi.sort_values(ascending=False)
#bscore = round(bgg[0],2)
#print(f"The variable with the biggest mutual info score is {bgg.index[0]}: {bscore}")

#I guess I should drop the above_average variable! Then add it to the y

# training
X_train = df_train.drop(['price', 'above_average'], axis=1)
y_train = pd.DataFrame(data=df_train['above_average'], columns=["above_average"])

# dev
X_val = df_val.drop(['price', 'above_average'], axis=1)
y_val = pd.DataFrame(data=df_val['above_average'], columns=["above_average"])

# test
X_test = df_test.drop(['price', 'above_average'], axis=1)
y_test = pd.DataFrame(data=df_test['above_average'], columns=["above_average"])


##  hot encoding

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical_columns + numerical_columns].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical_columns + numerical_columns].to_dict(orient='records')
X_val = dv.transform(val_dict)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#model = LogisticRegression(C=1.0, random_state=42,max_iter=1000)
num_iters = [5, 10, 100, 1000, 5000]
for num_iter in num_iters:
    lr = LogisticRegression(C=1.0, solver='liblinear', max_iter=num_iter, random_state=42) #this seems to be faster
    lr.fit(X_train, y_train['above_average'])
    y_hat_val= lr.predict(X_val)
    cur_accuracy = accuracy_score(y_val['above_average'].to_numpy(), y_hat_val)
    print(f"log regression model with {num_iter} iterarions gave this accuracy: {cur_accuracy}")

#choosing 100 iterations
#Which of following feature has the smallest difference?
#   * `neighbourhood_group`
#   * `room_type`
#   * `number_of_reviews`
#   * `reviews_per_month`
# ['neighbourhood_group', 'room_type']
# 7 numerical columns: ['latitude', 'longitude', 'reviews_per_month', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']

for dropcol in ['neighbourhood_group', 'room_type','reviews_per_month', 'number_of_reviews']:
    all_cols = categorical_columns + numerical_columns
    all_cols.remove(dropcol)
    train_dict = df_train[all_cols].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val[all_cols].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    lr = LogisticRegression(C=1.0, solver='liblinear', max_iter=100, random_state=42) #this seems to be faster
    lr.fit(X_train, y_train['above_average'])
    y_hat_val= lr.predict(X_val)
    cur_accuracy = accuracy_score(y_val['above_average'].to_numpy(), y_hat_val)
    print(f"log regression model without {dropcol} gave this accuracy: {cur_accuracy}")

