#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=0_mGopGP2dQ




#Dealing with null data
    #Recognizing blank data, that is automatically filled as Nan by Pandas

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')

print(df.isnull())#Shows whether a cell is null or not, not that helpful.
#Drop the entire column, if it makes sense
df = df.drop("Manual2", axis=1)
print(df.isnull().sum())   #Shows number of nulls in each column.

#If we only have handful of rows of null we can afford to drop these rows.
df2 = df.dropna()  #Drops all rows with at least one null value. 
#We can overwrite original df by equating it to df instead of df2.
#Or adding inplace=True inside
print(df2.head(25))  #See if null rows are gone.e.g. row 12

#If we have a lot of missing data then removing rows or columns
#may not be preferable.
#In such cases data scientists use Imputation technique.
#Just a fancy way of saying, fill it with whatever value
#A good guess would be filling missing values by the mean of the dataset.

print(df['Manual'].describe())  #Mean value of this column is 100.

df['Manual'].fillna(100, inplace=True)
print(df.head(25))   #Notice last entry in MinIntensity filled with 159.8


#In this example a better way to fill NaN is by filling with average of all auto columns from same row
import pandas as pd
import numpy as np

df = pd.read_csv('manual_vs_auto.csv')

df['Manual'] = df.apply(lambda row: (round((row['Auto_th_2']+row['Auto_th_3']+row['Auto_th_3'])/3)) if np.isnan(row['Manual']) else row['Manual'], axis=1)
print(df.head(25))
