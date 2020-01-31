#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=uBa7xSLy8V0


"""
@author: Sreenivas Bhattiprolu
"""

####################################

#Sorting Data
#####################################

#Sorting data
#We can sort rows using any of the columns. 
import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
print(df.sort_values('Manual', ascending=True))  #Small to large value
#Let us assign this to a diferent variable
df2=df.sort_values('Manual', ascending=True)
#We can select just a subset of data, for example to only get Manual column
print(df2['Manual'])


#To get multiple columns, it is just
print(df[['Manual', 'Auto_th_2']])

#To select subset of rows
print(df[20:30])  #Extracts rows 20 to 30, not including 30.

#Combining above two, to get specific columns from specific rows.
print(df.loc[20:30, ['Manual', 'Auto_th_2']])

#Selecting rows using row values, for example if we only want Set 2 info
#Similar to dropping rows we saw earlier.
set2_df = df[df['Unnamed: 0'] == 'Set2']
print(set2_df.tail())

print(max(df['Manual']))
#Instead of selection we can do data filtering,
#e.g. filter all values greater than certain size

print(df['Manual'] > 100.)  #Prints True or False.

#If we want to extract all data with this condition then use square brackets.
print(df[df['Manual'] > 100.])

#We can give multiple conditions to filter
print(df[(df['Manual'] > 100.) & (df['Auto_th_2'] < 100.)])

#We can use for loop to iterate just like we do for lists.
#Let's iterate through Auto, add them and divide by 3 to get average
#and compare with Manual value.


import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
for index, row in df.iterrows():
    average_auto = (row['Auto_th_2'] + row['Auto_th_3'] + row['Auto_th_4'])/3
    print(round(average_auto), row['Manual'])  #ROunding to INT for easy comparison

