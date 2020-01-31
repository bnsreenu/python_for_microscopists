#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=w1Tw7lj33yU




#  PART 2: SELECTING AND MANIPULATING DATA


#VIDEO: Deleting Rows and COlumns

##############
#Deleting columns
#Delete Manual2 column
import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')

df1 = df.drop("Manual2", axis=1) #Creating a new dataframe df1. 
# Axis=1 means referring to column. 
print(df.columns)
print(df1.columns)

#To drop multiple columns
df2=df.drop(["Manual2", "Auto_th_2", "Auto_th_3"], axis=1)
print(df2.columns)

#Inserting new columns, 

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
#as easy as just typing...
df['Date'] = "2019-05-06" 

print(df.head())  #New column addded
#But if you look at the data type....
print(df.dtypes)  #Date is not in date format, it is as object, otherwise string

#To properly format it as date so you can plot it later....
df['Date'] = pd.to_datetime("2019-05-06")

print(df.head())
print(df.dtypes)

#You can write the data back to a new csv.
df.to_csv('maual_vs_auto_updated.csv') #Open csv file to see

##################
#Deleting rows
import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')

#Delete a specific row
df1 = df.drop(df.index[1])
#Delete first 10 rows
print(df1.head())
df = df.iloc[10:,]
print(df.head())

#Drop all rows if the row value is equal to some string or number
df1 = df[df["Unnamed: 0"] != "Set1"]
print(df1.head())


