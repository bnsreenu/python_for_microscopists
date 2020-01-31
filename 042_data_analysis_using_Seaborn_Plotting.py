#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=ze7HGAf729k

####################################
#
#For better control over plotting you may as well use Matplotlib or Seaborn
#For Seaborn look here

##########################################
#Seaborn builds on top of matplotlib to provide a richer out of the box environment. 
# https://seaborn.pydata.org/
#https://seaborn.pydata.org/examples/index.html   #Checkout for more examples


import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')

import seaborn as sns

##############
#Single variable (distribution histogram plots)
#sns.distplot(df['Manual'])  #Will fail as we have a few missing values.

#Let us fill missing values with a value of 100
df['Manual'].fillna(100, inplace=True)

sns.distplot(df['Manual'])   #The overlay over histogram is KDE plot (Kernel density distribution)

#KDE plots. Kernel density estimation.
#KDE is a way to estimate the probability density function of a continuous random variable.

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)

import seaborn as sns
sns.kdeplot(df['Manual'], shade=True)

## Add Multiple plots
sns.kdeplot(df['Auto_th_2'], shade=True)
sns.kdeplot(df['Auto_th_3'], shade=True)
sns.kdeplot(df['Auto_th_4'], shade=True)
###################


#Basic line plot
import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)

import seaborn as sns
sns.set(style='darkgrid')   #Adds a grid
sns.lineplot(x='Image', y='Manual', data=df, hue='Unnamed: 0')   #Simple line plot
#Hue tells seaborn how to color various subcategories, like our set in this example.


##############################            
#Scatter plots
import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)
import seaborn as sns

#Basic scatter plot
sns.jointplot(x="Manual", y="Auto_th_2", data=df)
#KDE plot, Kernel density estimation.
sns.jointplot(x="Manual", y="Auto_th_2", data=df, kind="kde")


#Relationship between each feature and another selected feature can be easily plotted
#using pariplot function in Seaborn

import pandas as pd
import seaborn as sns

df = pd.read_csv('manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)
print(df.columns)

#sns.pairplot(df, x_vars=["Auto_th_2", "Auto_th_3", "Auto_th_4"], y_vars="Manual")

#too small. Let us chage the size

sns.pairplot(df, x_vars=["Auto_th_2", "Auto_th_3", "Auto_th_4"], y_vars="Manual", size=6, aspect=0.75)


#Scatterplot with linear regression

import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
import seaborn as sns

sns.lmplot(x='Manual', y='Auto_th_2', data=df, hue='Image_set', order=1)  #Scatterplot with linear regression fit and 95% confidence interval

#If you want equation, not possible to display in seaborn but you can get it the
#regular way using scipy stats module. 
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Manual'],df['Auto_th_2'])
print(slope, intercept)

#filtered = df[df['FileName'] != 'images/grains\grains1.jpg']
#filtered = df['FileName']
#sns.lmplot(x="Area", y="MeanIntensity", data=df, hue="orientation", fit_reg=False, col='FileName', col_wrap=2)

#Swarm plots
#Let's use manual_vs_auto2 file that we generated earlier 
import pandas as pd
df = pd.read_csv('manual_vs_auto2.csv')
df['Manual'].fillna(100, inplace=True)
print(df.head())

import seaborn as sns

#sns.swarmplot(x = "Image_set", y="Manual", data = df, hue="cell_count_index")

#SPlit each category
sns.swarmplot(x = "Image_set", y="Manual", data = df, hue="cell_count_index", dodge=True)


##################
"""
we can utilise the pandas Corr() to find the correlation between each variable 
in the matrix and plot this using Seaborn’s Heatmap function, 
specifying the labels and the Heatmap colour range.

"""


import pandas as pd
df = pd.read_csv('manual_vs_auto.csv')
print(df.dtypes)
df['Manual'].fillna(100, inplace=True)
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
import seaborn as sns
corr = df.loc[:,df.dtypes == 'int64'].corr() #Correlates all int64 columns
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
##########################
















