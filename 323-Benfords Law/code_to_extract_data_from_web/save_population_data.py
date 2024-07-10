"""
Code to scrape tabular data from a web page.


"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Define the URL of the web page you want to scrape for population data
population_url = 'https://www.worldometers.info/world-population/population-by-country/'

# Send a GET request to the population URL
population_response = requests.get(population_url)

# Parse the HTML content for population data using BeautifulSoup
population_soup = BeautifulSoup(population_response.text, 'html.parser')

# Find the table with the id "example2" for population data
population_table = population_soup.find("table", {"id": "example2"})

# Extract population data from the table rows
population_data = []

# Iterate over the rows in the population table
for row in population_table.find_all("tr"):
    columns = row.find_all("td")
    if len(columns) > 1:
        # Extract the relevant data (e.g., Country and Population)
        country = columns[1].text.strip()
        population = columns[2].text.strip()
        
        # Append the data to the population_data list
        population_data.append((country, population))

# Create a DataFrame for population data
population_df = pd.DataFrame(population_data, columns=['Country', 'Population'])

# Save the population data to a CSV file
population_df.to_csv('population_data.csv', index=False)

# Print the first few rows of the population DataFrame
print(population_df.head())
