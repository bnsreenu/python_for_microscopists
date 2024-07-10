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

In this python code, we analyze the distribution of leading digits in COVID cases 
and deaths data, with the objective of verifying whether the data adheres to Benford's Law. 

The observed frequencies of the leading digits are computed and compared against 
the expected frequencies predicted by Benford's Law. 


"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the COVID Data from the CSV file
covid_df = pd.read_csv("data/covid_data.csv")

# Verify Benford's Law for Total Cases
total_cases_digits = [
    int(str(cases).replace(",", "")[0]) for cases in covid_df["Total Cases"]
]
observed_counts_total_cases = [
    total_cases_digits.count(digit) for digit in range(1, 10)
]
total_records_total_cases = len(total_cases_digits)
expected_counts_total_cases = [
    int(total_records_total_cases * np.log10(1 + 1 / digit)) for digit in range(1, 10)
]

# Verify Benford's Law for Total Deaths, handling 'nan' values
total_deaths_digits = [
    int(str(deaths).replace(",", "")[0])
    if str(deaths).replace(",", "")[0] != "n"
    else None
    for deaths in covid_df["Total Deaths"]
]
observed_counts_total_deaths = [
    total_deaths_digits.count(digit) for digit in range(1, 10)
]
total_records_total_deaths = len([d for d in total_deaths_digits if d is not None])
expected_counts_total_deaths = [
    int(total_records_total_deaths * np.log10(1 + 1 / digit)) for digit in range(1, 10)
]

# Define the digits (1 through 9) for the x-axis
digits = range(1, 10)

# Create a line plot for observed frequencies (Total Cases)
sns.lineplot(x=digits, y=observed_counts_total_cases, label="Observed (Total Cases)")

# Create a line plot for expected frequencies (Total Cases)
sns.lineplot(x=digits, y=expected_counts_total_cases, label="Expected (Total Cases)")
# Add labels and a legend
plt.xlabel("Leading Digit")
plt.ylabel("Frequency")
plt.title("Leading Digit Distribution (COVID Data)")
plt.legend()

# Show the plot
plt.show()


# Create a line plot for observed frequencies (Total Deaths)
sns.lineplot(x=digits, y=observed_counts_total_deaths, label="Observed (Total Deaths)")
# Create a line plot for expected frequencies (Total Deaths)
sns.lineplot(x=digits, y=expected_counts_total_deaths, label="Expected (Total Deaths)")

# Add labels and a legend
plt.xlabel("Leading Digit")
plt.ylabel("Frequency")
plt.title("Leading Digit Distribution (COVID Data)")
plt.legend()

# Show the plot
plt.show()
