"""
Code to scrape tabular data from a web page.


"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the web page you want to scrape
url = 'https://www.worldometers.info/coronavirus/'

# Send a GET request to the URL
response = requests.get(url)

# Assuming that response.text contains the HTML response
html_text = response.text

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_text, 'html.parser')

# Find the table with ID "main_table_countries_today"
#We got this information by inspecting the web page (Right-click and then inspect)
covid_table = soup.find("table", {"id": "main_table_countries_today"})

# Extract data from the table rows
covid_data = []

# Iterate over the rows in the table
for row in covid_table.find_all("tr")[1:]:  # Skip the header row
    columns = row.find_all("td")
    if len(columns) > 1:
        # Extract the relevant data (e.g., Country, Total Cases, New Cases, etc.)
        country = columns[1].text.strip()
        total_cases = columns[2].text.strip()
        new_cases = columns[3].text.strip()
        total_deaths = columns[4].text.strip()
        new_deaths = columns[5].text.strip()
        
        # Append the data to the covid_data list
        covid_data.append((country, total_cases, new_cases, total_deaths, new_deaths))

# Create a DataFrame from the extracted data
covid_df = pd.DataFrame(covid_data, columns=["Country", "Total Cases", "New Cases", "Total Deaths", "New Deaths"])

# Save the DataFrame to a CSV file
covid_df.to_csv("covid_data.csv", index=False)

# Print the first few rows of the DataFrame
print(covid_df.head())
