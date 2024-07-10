"""
Code to scrape tabular data from a web page.


"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Define the URL of the web page you want to scrape for GDP data
gdp_url = 'https://www.worldometers.info/gdp/gdp-by-country/'

# Send a GET request to the GDP URL
gdp_response = requests.get(gdp_url)

# Parse the HTML content for GDP data using BeautifulSoup
gdp_soup = BeautifulSoup(gdp_response.text, 'html.parser')

# Find the table with the id "example2" for GDP data
gdp_table = gdp_soup.find("table", {"id": "example2"})

# Extract GDP data from the table rows
gdp_data = []

# Iterate over the rows in the GDP table
for row in gdp_table.find_all("tr"):
    columns = row.find_all("td")
    if len(columns) > 1:
        # Extract the relevant data (e.g., Country and GDP)
        country = columns[1].text.strip()
        gdp = columns[2].text.strip()
        
        # Use regular expressions to remove non-numeric characters and convert to float
        gdp_value = re.sub('[^\d.]', '', gdp)
        gdp_float = float(gdp_value) if gdp_value else 0.0  # Handle empty strings
        
        # Append the data to the gdp_data list
        gdp_data.append((country, gdp_float))

# Create a DataFrame for GDP data
gdp_df = pd.DataFrame(gdp_data, columns=['Country', 'GDP (nominal, 2022)'])

# Save the GDP data to a CSV file
gdp_df.to_csv('gdp_data.csv', index=False)

# Print the first few rows of the GDP DataFrame
print(gdp_df.head())
