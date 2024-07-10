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

In this python code, we analyze the distribution of leading digits in tax deduction 
amounts, with the objective of verifying whether the data adheres to Benford's Law. 

The observed frequencies of the leading digits are computed and compared against 
the expected frequencies predicted by Benford's Law. 


"""


import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load tax return data from CSV
csv_file = "data/synthetic_tax_return_data.csv"

# Initialize a list to store deduction amounts
deduction_amounts = []

# Read the CSV file and extract deduction amounts
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        deduction = row[4].replace('$', '').replace(',', '').strip()
        if deduction:
            deduction_amounts.append(float(deduction))

# Calculate the leading digits for deduction amounts
leading_digits = [int(str(amount)[0]) for amount in deduction_amounts if amount > 0]

# Calculate observed frequencies (counts of leading digits)
observed_counts = [leading_digits.count(digit) for digit in range(1, 10)]

# Calculate the expected frequencies according to Benford's Law
expected_counts = [int(len(leading_digits) * np.log10(1 + 1/digit)) for digit in range(1, 10)]

# Define the digits (1 through 9) for the x-axis
digits = range(1, 10)

# Create a line plot for observed frequencies
sns.lineplot(x=digits, y=observed_counts, label='Observed')

# Create a line plot for expected frequencies
sns.lineplot(x=digits, y=expected_counts, label='Expected')

# Add labels and a legend
plt.xlabel('Leading Digit')
plt.ylabel('Frequency')
plt.title('Leading Digit Distribution for Deductions (Observed vs. Expected)')
plt.legend()

# Show the plot
plt.show()
