import csv

# Define the synthetic tax return data
tax_return_data = [
    ["Name", "Income Source", "Income", "Deduction Type", "Deduction Amount"],
    ["John Doe", "Salary", "$50,000.00", "Mortgage Interest", "$1,200.00"],
    ["", "Freelance Income", "$7,500.00", "Property Taxes", "$900.00"],
    ["", "Rental Property 1", "$12,000.00", "State Income Tax", "$2,000.00"],
    ["", "Investment Dividends", "$2,500.00", "Health Insurance", "$1,200.00"],
    ["", "Business Income", "$10,000.00", "Charitable Donations", "$500.00"],
    ["", "", "", "Home Office Expense", "$600.00"],
    ["", "", "", "Education Expenses", "$1,800.00"],
    ["", "", "", "Business Travel", "$850.00"],
    ["", "", "", "Professional Fees", "$1,200.00"],
    ["", "", "", "Utilities", "$750.00"],
    ["", "", "", "Transportation", "$1,000.00"],
    ["", "", "", "Entertainment", "$400.00"],
    ["", "", "", "Medical Expenses", "$2,500.00"],
    ["", "", "", "Charitable Donations", "$700.00"],
    ["", "", "", "Other Deductions", "$1,200.00"]
]

# Specify the CSV file name
csv_file = "data/synthetic_tax_return_data.csv"

# Open the CSV file in write mode and write the data
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(tax_return_data)

print(f"Tax return data has been saved to {csv_file}")
