"""
Python's Hidden Gems: Uncovering Lesser-Known Tricks
Simplifying Code with defaultdict

You start with an empty notebook, and you want to add three types of fruits: 
    apples, bananas, and cherries.

To do this, you open the notebook and look for the page for "apples." 
But the notebook is empty, so there's no page for "apples."

You have to create a page for "apples" first and write down that you have 2 apples.

Next, you want to add bananas. You look for the page for "bananas," 
but it's not there, so you create a new page for "bananas" and write down that 
you have 1 banana.

You repeat this process for cherries, creating a page for them and writing 
that you have 3 cherries.

Now, imagine someone asks you, "How many dates do you have?" You try to look 
for a "dates" page in your notebook, but it doesn't exist.

You have to be careful not to make mistakes and accidentally forget to create 
a page for any fruit or forget to update the count correctly. 
This manual checking and updating make your notebook longer and more error-prone.
"""


fruit_counts = {}

# Add some fruits and their counts
fruit = 'apple'
if fruit in fruit_counts:
    fruit_counts[fruit] += 2
else:
    fruit_counts[fruit] = 2

fruit = 'banana'
if fruit in fruit_counts:
    fruit_counts[fruit] += 1
else:
    fruit_counts[fruit] = 1

fruit = 'cherry'
if fruit in fruit_counts:
    fruit_counts[fruit] += 3
else:
    fruit_counts[fruit] = 3

fruit = 'date'
if fruit in fruit_counts:
    fruit_counts[fruit] += 0
else:
    fruit_counts[fruit] = 0

# Accessing a non-existing key requires similar checks
fruit = 'date'
if fruit in fruit_counts:
    print(fruit_counts[fruit])
else:
    print(0)

print(fruit_counts)


#########################
"""
You start with a magical notebook (a defaultdict) that automatically creates 
a page for any fruit you mention.

You confidently write down the counts of your fruits without worrying if the 
pages exist or not. For example, you just say, "I have 2 apples," 
and the notebook creates an "apples" page for you if it doesn't already exist.

You add bananas and cherries the same way, without needing to manually check 
or create pages.

Now, when someone asks about dates, you confidently check the "dates" page in 
your notebook. If it doesn't exist, the notebook kindly tells you that you 
have 0 dates.

Your magical notebook simplifies your record-keeping. You don't have to write 
extra code to handle missing pages or worry about making mistakes, 
making your record-keeping shorter and more accurate.

"""

from collections import defaultdict

# Create a defaultdict with an int default factory
fruit_counts = defaultdict(int)

# Add some fruits and their counts
fruit_counts['apple'] += 2
fruit_counts['banana'] += 1
fruit_counts['cherry'] += 3

# Accessing a non-existing key returns the default value (0 for int)
print(fruit_counts['date'])  # Output: 0

# You can use the defaultdict just like a regular dictionary
print(fruit_counts)  
