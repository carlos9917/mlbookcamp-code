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

del df_train['price']
del df_val['price']
del df_test['price']

categorical_columns = list(df[base].dtypes[df[base].dtypes == 'object'].index)
#print(categorical_columns)
#numerical_columns = df_train.columns - categorical_columns
numerical_columns = list(df[base].dtypes[df[base].dtypes == 'float'].index) + list(df[base].dtypes[df[base].dtypes == 'int'].index)
numerical_columns.remove("price")
print(f"{len(categorical_columns)} categorical columns: {categorical_columns}")
print(f"{len(numerical_columns)} numerical columns: {numerical_columns}")

corrs=df_full_train[categorical_columns + numerical_columns].corrwith(df_full_train.price).to_frame('correlation')
print("Vars with biggest correlation")
print(corrs)
#df[x].corrwith(y)

##
##Q3
# create variable above_average, which is 1 if price above 152
above_average=[]
#binarizing the price
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

train_dict = df_train[categorical_columns].to_dict(orient='records')
#train_dict = df_train[categorical_columns + numerical_columns].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical_columns].to_dict(orient='records')
#val_dict = df_val[categorical_columns + numerical_columns].to_dict(orient='records')
X_val = dv.transform(val_dict)

#Using all the columns this does not work at all. If I select only the categorical
# I am able to move forward
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, random_state=42,max_iter=1000)


def use_log_reg():
    # solver='lbfgs' is the default solver in newer version of sklearn
    # for older versions, you need to specify it explicitly
    model.fit(X_train, y_train)
    
    #y_pred = model.predict_proba(X_val)[:, 1]
    
    #y_pred = model.predict(X_val) #[:, 1]
    #price_decision = (y_pred >= 152 ) #0.5)
    #accu = (y_val == y_pred).mean()
    #accu = (y_val == price_decision).mean()
    y_pred = model.predict(X_val)
    score = round(model.score(X_val,y_val),2)
    #not sure why getting 0.069 and not 0.69 here!!!
    print(f'Accuracy of logistic regression  classifier on test set: {score}')
    #print(f"Price accuracy {model.score(x_test,)}")

### Question 5
#use_log_reg()
for feature in df_train.columns:
    print(f"Dropping {feature} from X_train")
    df.drop(columns[feature],inplace=True)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    y_train = df_train.price.values
    y_val = df_val.price.values
    y_test = df_test.price.values
    
    del df_train['price']
    del df_val['price']
    del df_test['price']

    use_log_reg()
    # 
"""

* We have 9 features: 7 numerical features and 2 categorical.
* Let's find the least useful one using the *feature elimination* technique.
* Train a model with all these features (using the same parameters as in Q4).
* Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
* For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
* Which of following feature has the smallest difference?
   * `neighbourhood_group`
   * `room_type`
   * `number_of_reviews`
   * `reviews_per_month`

> **note**: the difference doesn't have to be positive
"""



### Question 6
"""

* For this question, we'll see how to use a linear regression model from Scikit-Learn
* We'll need to use the original column `'price'`. Apply the logarithmic transformation to this column.
* Fit the Ridge regression model on the training data.
* This model has a parameter `alpha`. Let's try the following values: `[0, 0.01, 0.1, 1, 10]`
* Which of these alphas leads to the best RMSE on the validation set? Round your RMSE scores to 3 decimal digits.

If there are multiple options, select the smallest `alpha`.

"""
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


#Apply the log transformation to the price variable using the np.log1p() function.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42) #note two splits, here 20 % of 80% is 25%
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)
del df_train['price']
del df_val['price']
del df_test['price']

from sklearn.linear_model import Ridge
for alpha in [0, 0.01, 0.1, 1, 10]:
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_val) #[:, 1]
    #reg = LinearRegression().fit(X_train, y_train)
    #print(reg.score(X_val, y_val))
    #w_0, w = rutils.train_linear_regression_reg(X_train, y_train,r)
    #y_pred = w_0 + X_train.dot(w)
    score = round(rmse(y_val, y_pred),3)
    print(f'validation using alpha={alpha}: {score}')

