#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/29YnoZcrW3o

#To download the file locally
#import urllib
#url = "https://covid.ourworldindata.org/data/ecdc/full_data.csv"
#urllib.request.urlretrieve (url, "data/full_data.csv")

import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib

#date	location	new_cases	new_deaths	total_cases	total_deaths
CVD = pd.read_csv('https://covid.ourworldindata.org/data/ecdc/full_data.csv')
print(CVD.head())
print(CVD.dtypes)

#dateFormat = '%Y-%m-%d'
# Convert string values of date to datetime format
CVD['date'] = [dt.datetime.strptime(x,'%Y-%m-%d') for x in CVD['date']] 
#print(CVD.dtypes)

#Let's look at multiple countries
countries=['United States', 'Spain', 'Italy']
CVD_country = CVD[CVD.location.isin(countries)]  #Create subset data frame for select countries


CVD_country.set_index('date', inplace=True)  #Make date the index for easy plotting

#To create subset range based on dates
#CVD_country = CVD_country.loc['2020-02-15':'2020-03-22']

#print(CVD_country.tail())  #Check the last date 

#To calculate mortality rate
CVD_country['mortality_rate'] = CVD_country['total_deaths']/CVD_country['total_cases']
#print(CVD_country.tail())

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

CVD_country.groupby('location')['new_cases'].plot(ax=axes[0,0], legend=True) #for log scale add logy=True
CVD_country.groupby('location')['new_deaths'].plot(ax=axes[0,1], legend=True)
CVD_country.groupby('location')['total_cases'].plot(ax=axes[1,0], legend=True)
CVD_country.groupby('location')['total_deaths'].plot(ax=axes[1,1], legend=True)
#CVD_country.groupby('location')['mortality_rate'].plot(ax=axes[1,1], legend=True)
#CVD_country.to_csv('data/output.csv')

axes[0, 0].set_title("New Cases")
axes[0, 1].set_title("New Deaths")
axes[1, 0].set_title("Total Cases")
axes[1, 1].set_title("Total Deaths")

fig.tight_layout()  # adjust subplot parameters to give specified padding.