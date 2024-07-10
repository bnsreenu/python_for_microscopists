"""
Benford's Law, also known as the first-digit law, is a statistical phenomenon 
observed in many sets of numerical data. It states that in certain naturally 
occurring datasets, the leading digits (1, 2, 3, etc.) occur with a higher frequency 
than larger digits (4, 5, 6, etc.). According to Benford's Law, the distribution 
of leading digits follows a logarithmic pattern, where smaller digits are more 
likely to be the first digit in a number. This surprising and counterintuitive 
property is frequently encountered in diverse datasets such as financial transactions,
population numbers, and scientific data, making Benford's Law a useful tool for 
detecting anomalies and irregularities in numerical datasets.

In this python code, we analyze the distribution of leading digits in national GDP 
amounts, with the objective of verifying whether the data adheres to Benford's Law. 

The observed frequencies of the leading digits are computed and compared against 
the expected frequencies predicted by Benford's Law. 


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the GDP Data from the CSV file
gdp_df = pd.read_csv("data/gdp_data.csv")

# Extract the leading digit for each GDP value and store it in a list
leading_digits = [int(str(gdp).replace(',', '')[0]) for gdp in gdp_df['GDP (nominal, 2022)']]

# Calculate observed frequencies (counts of leading digits)
observed_counts = [leading_digits.count(digit) for digit in range(1, 10)]

# Calculate the expected frequencies according to Benford's Law
total_records = len(leading_digits)
expected_counts = [int(total_records * np.log10(1 + 1/digit)) for digit in range(1, 10)]

# Define the digits (1 through 9) for the x-axis
digits = range(1, 10)

# Create a line plot for observed frequencies
sns.lineplot(x=digits, y=observed_counts, label='Observed')

# Create a line plot for expected frequencies
sns.lineplot(x=digits, y=expected_counts, label='Expected')

# Add labels and a legend
plt.xlabel('Leading Digit')
plt.ylabel('Frequency')
plt.title('Leading Digit Distribution (GDP Data)')
plt.legend()

# Show the plot
plt.show()
