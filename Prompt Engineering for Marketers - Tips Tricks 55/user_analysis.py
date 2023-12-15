# Purpose: Analyzing user sign-ups for a SaaS platform using data from a CSV file and generating plots.

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
file_path = 'users.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to understand the structure of the data
print("Data Overview:")
print(df.head())

# Task 1: Number of users for each country
# Plot a bar chart
plt.figure(figsize=(12, 6))
df['Country'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Number of Users for Each Country')
plt.xlabel('Country')
plt.ylabel('Number of Users')
plt.show()

# Task 2: Total revenue per country
# Convert Subscription_Price to numeric (remove '$' and convert to float)
df['Subscription_Price'] = df['Subscription_Price'].replace('[\$,]', '', regex=True).astype(float)

# Group by country and calculate total revenue
revenue_per_country = df.groupby('Country')['Subscription_Price'].sum()

# Plot a bar chart
plt.figure(figsize=(12, 6))
revenue_per_country.sort_values(ascending=False).plot(kind='bar', color='green')
plt.title('Total Revenue per Country')
plt.xlabel('Country')
plt.ylabel('Total Revenue ($)')
plt.show()

# Task 3: Subscription type for each country
# Group by country and subscription type, then calculate the count
subscription_type_per_country = df.groupby(['Country', 'Subscription_Price']).size().unstack(fill_value=0)

# Plot a stacked bar chart
plt.figure(figsize=(14, 7))
subscription_type_per_country.plot(kind='bar', stacked=True)
plt.title('Subscription Type for Each Country')
plt.xlabel('Country')
plt.ylabel('Number of Users')
plt.legend(title='Subscription Price', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
