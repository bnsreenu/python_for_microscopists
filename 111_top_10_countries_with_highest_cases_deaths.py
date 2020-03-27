#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"


#https://youtu.be/g_MaZK3x1Vk

"""
What are the top 10 countries with highest COVID cases and deaths. 
Dataset location: 
    https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
Based on the blog by: 
    https://blog.rmotr.com/learn-data-science-by-analyzing-covid-19-27a063d7f442

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
from plotly.offline import plot  #To plot in Spyder IDE. Opens plots in the default browser

#Breaking the numbers down for each country
#What are the countries with highest cases, recoveries and mortality rates?
#For plotting we need numbers arraged in the right format: 
# Country/Region, total Confirmed, deaths, recovered, active as of today. 

covid_confirmed   = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
covid_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
covid_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


print(covid_confirmed.shape)
print(covid_deaths.shape)
print(covid_recovered.shape)


# Start by converting all data into the long format

covid_confirmed_long = pd.melt(covid_confirmed,
                               id_vars=covid_confirmed.iloc[:, :4],
                               var_name='date',
                               value_name='confirmed')

covid_deaths_long = pd.melt(covid_deaths,
                               id_vars=covid_deaths.iloc[:, :4],
                               var_name='date',
                               value_name='deaths')

covid_recovered_long = pd.melt(covid_recovered,
                               id_vars=covid_recovered.iloc[:, :4],
                               var_name='date',
                               value_name='recovered')

print(covid_confirmed_long.head())

#Merge all three dataframes into one for easy plotting
covid_df = covid_confirmed_long
covid_df['deaths'] = covid_deaths_long['deaths']
covid_df['recovered'] = covid_recovered_long['recovered']

print(covid_df.head())

#Add a new column for active cases
covid_df['active'] = covid_df['confirmed'] - covid_df['deaths'] - covid_df['recovered']
print(covid_df.head())

#Since we are reading raw csv files again, let us clean up the data
#Replace mainland china by china and fill null values with 0

covid_df['Country/Region'].replace('Mainland China', 'China', inplace=True)
covid_df[['Province/State']] = covid_df[['Province/State']].fillna('')
covid_df.fillna(0, inplace=True)
print(covid_df.isna().sum().sum())

#Save the data as csv to local drive
covid_df.to_csv('covid_df.csv', index=None) #optional but would be nice to have a local copy

#if saved to local drive reload it.
#pd.read_csv('covid_df.csv')

#Aggregate data by Country/Region and then Province/State
#Find out maximum values as a function of time
covid_countries_df = covid_df.groupby(['Country/Region', 'Province/State']).max().reset_index()

#Group the data by Country/Region, get sum of cases every state in the country.
covid_countries_df = covid_countries_df.groupby('Country/Region').sum().reset_index()

#Remove Lat and Long columns as we would not be using them
covid_countries_df.drop(['Lat', 'Long'], axis=1, inplace=True)

print(covid_countries_df)

# DATA is READY to be plotted.

#Top 10 countries with confirmed cases
top_10_confirmed = covid_countries_df.sort_values(by='confirmed', ascending=False).head(10)

fig = px.bar(top_10_confirmed.sort_values(by='confirmed', ascending=True),
             x="confirmed", y="Country/Region",
             title='Confirmed Cases', text='confirmed',
             template='plotly_dark', orientation='h')

fig.update_traces(marker_color='#3498db', textposition='outside')

plot(fig)

#Top 10 countries with high recovery numbers
top_10_recovered = covid_countries_df.sort_values(by='recovered', ascending=False).head(10)
fig = px.bar(top_10_recovered.sort_values(by='recovered', ascending=True),
             x="recovered", y="Country/Region",
             title='Recovered Cases', text='recovered',
             template='plotly_dark', orientation='h')

fig.update_traces(marker_color='#2ecc71', textposition='outside')

plot(fig)

#Top 10 countries with highest number of deaths
top_10_deaths = covid_countries_df.sort_values(by='deaths', ascending=False).head(10)
fig = px.bar(top_10_confirmed.sort_values(by='deaths', ascending=True),
             x="deaths", y="Country/Region",
             title='Death Cases', text='deaths',
             template='plotly_dark', orientation='h')

fig.update_traces(marker_color='#e74c3c', textposition='outside')
plot(fig)

#Top 10 countries with highest mortality rate
covid_countries_df['mortality_rate'] = round(covid_countries_df['deaths'] / covid_countries_df['confirmed'] * 100, 2)
temp = covid_countries_df[covid_countries_df['confirmed'] > 100]
top_10_mortality_rate = temp.sort_values(by='mortality_rate', ascending=False).head(10)

fig = px.bar(top_10_mortality_rate.sort_values(by='mortality_rate', ascending=True),
             x="mortality_rate", y="Country/Region",
             title='Mortality rate', text='mortality_rate',
             template='plotly_dark', orientation='h',
             width=700, height=600)

fig.update_traces(marker_color='#c0392b', textposition='outside')

plot(fig)

