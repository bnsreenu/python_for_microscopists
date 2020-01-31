#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=p7KyukHE9xU


##########################################
#Using group by 
#Group-by’s can be used to build groups of rows based off a specific feature 
#eg. the Set name in our csv dataset, we can group by set 1, 2, 3, and 4
#We can then perform an operation such as mean, min, max, std on the individual groups    

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
#Let us rename Unnamed column and drop Manual 2 column
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
df = df.drop("Manual2", axis=1)
print(df.head())
group_by_file = df.groupby(by=['Image_set'])
set_data_count = group_by_file.count()  #Count for each value per group
set_data_avg = group_by_file.mean()  #Mean for each value per group
print(set_data_count)
print(set_data_avg)

#Correlation between data
print(df.corr())  #Correlation between all columns



#To check correlation for specific columns
import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
print(df['Manual'].corr(df['Auto_th_2'])) 

"""
Positive numbers indicate a positive correlation — one goes up 
the other goes up — and negative numbers represent an inverse correlation — 
one goes up the other goes down. 1.0 indicates a perfect correlation.

"""
