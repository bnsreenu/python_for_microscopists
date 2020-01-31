#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=QKaJO8I5YaM



"""
Pandas is a python library that enables easy data analysis.
Think of Pandas as an extension of Numpy where it makes handling of arrays
easy, almost like Excel.

PART 1: LOADING, VIEWING AND  UNDERSTANGING DATA

"""
#Once you get the csv file, ploting and analyzing is very easy with Pands library. 
#Here, just 3 lines to get our plot. 
import pandas as pd

df = pd.read_csv('images/grains/grain_measurements.csv')

df['Area'].plot(kind='hist', title='Area', bins=50)

#The basics of Pandas.

#Pandas library is popular package for datascience because it makes it easy to work with data.
#It makes data manipulation and analysis easy. 
#Pandas handle data as dataframe, 2D datastructure, meaning, data is like tabular form, columns and rows

#############################

#Now let us pass data directly.
import pandas as pd
data = [[10, 200, 60],
        [12, 155, 45],
        [9, 50, -45.],
        [16, 240, 90]] 
         
df = pd.DataFrame(data, index = [1,2,3,4], columns = ['Area', 'Intensity', 'Orientation'])
print(df)

###########################################
#Now let us load data from a text or csv file
#Dataset showing Different image sets (25 images in each set)
#Total 100 images analyzed manually and automatically using cell counting algorithm
#Each image was manually analyzed to count cells
#An attempt was made to count manually by a different person but gave up after first 3 images
#Then analyzed using the algorithm we developed earlier, 
#by changing a parameter 3 times, giving different cell counts each time 

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')

print(df.info())  #Prvides an overview of the dataframe. 
print(df.shape)  #How many rows and columns

print(df)  #Shows a lot of stuff but truncated
print(df.head(7))  #Default prints 5 rows from the top
#First default column you see are indices. 
print(df.tail())   #Default prints 5 rows from the bottom

#First line in csv is considered header, even if you don't specify
# so it prints it out every time
#First column is the index and it goes from 0, 1, 2, ....
#Index is not part of the data frame
#INdex is the unique identifier of a row, in our case a specific grain in a specific image
#Any of the other columns can be assigned as index if we know it is a unique identifier. 

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
print(df.index)  #Defines start and stop with step size. Not very exciting with default index
#But can be useful if we assign other column as index. 
df = df.set_index('Image')
print(df.head())
#View all column names.
print(df.columns)   #Image name column disappeared as it is used as index. 

#TO look at all unique entires. In this case, our 3 file names. 
print(df['Unnamed: 0'].unique())  

#If unnamed is bothering you then you can change the name.
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
print(df.columns) 
#Missing data is encoded as NaN so we can work with them in a friendly manner. 
#Let us look at Manual column to see what it has.
print(df["Manual"])  #Shows NAN. We can fill it with something or ignore it or remove the column
#Let us look at manipulating data in our next video. 

#For now let us finish by looking at a couple more useful functions. 
#Pandas automatically recognizes correct data types.

print(df.dtypes)  

"""

#Similarly multiple column names can be changed at once. 
df = df.rename(columns = {'equivalent_diameter':'Diameter(um)', 
                          'Area':'Area(sq. um)',
                          'orientation':'orientation (deg)',
                          'MajorAxisLength':'Length (um)',
                          'MinorAxisLength':'Width (um)',
                          'Perimeter':'Perimeter (um)'})
print(df.dtypes)
"""

print(df.describe())  #Gives statistical summary of each column. 

#######################################################################

