#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=Dx8ixwTfWz0


#Many ways to read and write CSV files in Python

#READING

import csv

with open('fruits1.csv', newline='') as myFile:  
    reader = csv.reader(myFile)
    for row in reader:
        print(row)

#WRITING
#Using core Python without any special libraries

some_list=['apples', 'oranges', 'grapes', 'mangoes']

output_file = open('fruits.csv', 'w')
for fruit in some_list:
    output_file.write(fruit)
    output_file.write('\n')



some_list=['apples', 'oranges', 'grapes', 'mangoes']
with open('fruits.csv', 'w') as f:
    f.write('\n'.join(some_list))


#Using CSV writer
import csv
some_list=[['apples', 'oranges', 'grapes', 'mangoes'], [20, 25, 42, 35]]

with open('fruits.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for fruit in some_list:
        writer.writerow(fruit)
        



import csv

some_list=[['apples', 'oranges', 'grapes', 'mangoes'], [20, 25, 42, 35]]
output_file = open('fruits.csv', 'w')
with output_file:  
   writer = csv.writer(output_file)
   writer.writerows(some_list)
   

#Adding header and rows..

import csv

header = ['name', 'age', 'ZIP']
rows = [['John', 30, 94568],
        ['Mary', 40, 94588],
        ['Henry', 50, 94566]]

with open('people.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)
    for row in rows:
        csv_writer.writerow(row)

#Using dictwriter
import csv      

header = ['name', 'age', 'ZIP']        
rows = [{'name':'John', 'age':30, 'ZIP':94568},
        {'name':'Mary', 'age':40, 'ZIP':94588},
        {'name':'Henry', 'age':50, 'ZIP':94566},
        ]

with open('people.csv', 'w') as f:
 
    csv_writer = csv.DictWriter(f, fieldnames=header) 
    csv_writer.writeheader() # write header 
    csv_writer.writerows(rows)

