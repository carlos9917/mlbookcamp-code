#Homework 3. Questions 1-6
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import H2.reg_utils as rutils
data = pd.read_csv("../H2/AB_NYC_2019.csv")
#from rich import print


base=['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']



#What is the most frequent observation (mode) for the column 'neighbourhood_group'?
sel_col="neighbourhood_group"
most_common = data[sel_col].mode()[0]
print(f"Most common observation in {sel_col}: {most_common}")
df = data.copy()


"""
Split the data
Split your data in train/val/test sets, with 60%/20%/20% distribution.
Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
Make sure that the target value ('price') is not in your dataframe.
"""
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

del df_train['price']
del df_val['price']
del df_test['price']

categorical_columns = list(df[base].dtypes[df[base].dtypes == 'object'].index)
#print(categorical_columns)
#numerical_columns = df_train.columns - categorical_columns
numerical_columns = list(df[base].dtypes[df[base].dtypes == 'float'].index) + list(df[base].dtypes[df[base].dtypes == 'int'].index)
numerical_columns.remove("price")
#print(f"Categorical columns: {categorical_columns}")
#print(f"Numerical columns: {numerical columns}")

#corrs=df_full_train[categorical_columns + numerical_columns].corrwith(df_full_train.price).to_frame('correlation')
#print("Vars with biggest correlation")
#print(corrs)
#df[x].corrwith(y)

##
##Q3
# create variable above_average, which is 1 if price above 152
above_average=[]
for k in df["price"].index:
    if df["price"].values[k] >= 152:
        above_average.append(k)
        df["price"].values[k] = 1
    else:
        df["price"].values[k] = 0

from sklearn.metrics import mutual_info_score
#Apply the mutual info score to price
#categorical_columns=["room_type","neighbourhood_group"]
def mutual_info_price_score(series):
    return mutual_info_score(series, df_full_train.price)
mi = df_full_train[categorical_columns].apply(mutual_info_price_score)
bgg=mi.sort_values(ascending=False)
bscore = round(bgg[0],2)
print(f"The variable with the biggest mutual info score is {bgg.index[0]}: {bscore}")
#mutual_info_score(series, df_train_full.)


# Q4 Logistic reg

## First do the hot encoding

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

#train_dict = df_train[categorical_columns + numerical_columns].to_dict(orient='records')
train_dict = df_train[categorical_columns].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical_columns].to_dict(orient='records')
X_val = dv.transform(val_dict)

from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver='lbfgs')
#model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model = LogisticRegression(C=1.0, random_state=42,max_iter=500)

# solver='lbfgs' is the default solver in newer version of sklearn
# for older versions, you need to specify it explicitly
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

