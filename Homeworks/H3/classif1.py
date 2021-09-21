#Homework 2. Questions 1-4
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import H2.reg_utils as rutils
data = pd.read_csv("../H2/AB_NYC_2019.csv")


base=['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']



#What is the most frequent observation (mode) for the column 'neighbourhood_group'?
sel_col="neighbourhood_group"
most_common = data[sel_col].mode()[0]
print(f"Most common observation in {sel_col}: {most_common}")


"""
Split the data
Split your data in train/val/test sets, with 60%/20%/20% distribution.
Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
Make sure that the target value ('price') is not in your dataframe.
"""

df = data.copy()
del df["price"]
#note, seed is set to 42 by default here
df_train,df_val,df_test = rutils.split_data(df)


"""

Question 2
Create the correlation matrix for the numerical features of your train dataset.
In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.
What are the two features that have the biggest correlation in this dataset?

"""
