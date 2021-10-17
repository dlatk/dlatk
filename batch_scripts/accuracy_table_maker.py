#!/usr/bin/python3
############################
## accuracy_table_maker.py
##
## given a list of accuracy csv files
## prints out a nice set of tables.
## (does a horizontal append, attempting to align rows)

## USAGE:   accuracy_table_maker.py CSV_FILE1.csv [MORE CSV FILES] [COLUMNS TO EXTRACT]
## EXAMPLE: accuracy_table_maker.py outcomes1.accuracy_data.csv outcomes2.accuracy_data.csv auc auc_p

import os
import sys
import csv
import numpy as np
import re

#constants
DEFAULT_COLUMNS = ['auc', 'auc_p']
ROW_ID_COLUMN = 'model_controls' #'row_id' #id to use as the marker of the row in the original data

#GET files and columns
accuracy_files = []
columns = []
for arg in sys.argv[1:]:
    if arg[-4:].lower() == '.csv':
        accuracy_files.append(arg)
    else:
        columns.append(arg)

if len(accuracy_files) < 1:
    print("No accuracy files found in command.\n")
    print("USAGE:   accuracy_table_maker.py CSV_FILE1.csv [MORE CSV FILES] [COLUMNS TO EXTRACT]")
    print("EXAMPLE: accuracy_table_maker.py outcomes1.accuracy_data.csv outcomes2.accuracy_data.csv auc auc_p")
    print("NOTE: all csv files must end in .csv")
    sys.exit(1)
if len(columns) < 1:
    columns = DEFAULT_COLUMNS

def parseAccFile(filename, columns, row_id = ROW_ID_COLUMN):
    #returns all rows of the given columns for the file
    #results['row_id']['column'] = value
    results = dict()
    
    with open(filename) as f:
        #read until finding header line with columns

        header = []
        row_ids = set()
        while not header:
            line = f.readline()
            if (line==''):
                print("\nNever found columns", columns, "in a header line.")
                sys.exit(1)
            if np.sum([re.search(r'\b'+c+r'\b', line) != None for c in [row_id]+columns]) == (len(columns)+1):
                header = list(csv.reader([line]))[0]

        #now grab all of the columns per row:
        creader = csv.DictReader(f, fieldnames=header)
        for row in creader:
            #print(row)#DEBUG
            rid = row[row_id].replace(',', '')
            results[rid] = {c:row[c] for c in columns}
            row_ids.add(rid)

    return results, row_ids

print("\nREADING Files:")
results = []
row_ids = set()
for i in range(len(accuracy_files)):
    filename = accuracy_files[i]
    fnum = i + 1
    print(fnum, ": ", filename)

    rs, ids = parseAccFile(filename, columns)
    row_ids = row_ids | ids
    results.append(rs)
    #print(results[-1])
    
print("\nTABLE:\n")
#header
table_header = ', '.join(["%3s: %-15s" % (i+1, accuracy_files[i][-33:-18]) for i in range(len(accuracy_files))])
print("%18s,"% "", table_header)
col_len = int(18 / len(columns))
column_header = ', '.join(["%9s" % c[:col_len] for i in range(len(accuracy_files)) for c in columns])
print("%18s,"%ROW_ID_COLUMN[-15:], column_header)

#add results rows:
for row_id in sorted(row_ids):
    row = "%18s" % row_id[-18:]
    for i in range(len(accuracy_files)):
        try:
            r = results[i][row_id]
            row += ", " + ', '.join([" "*(col_len-5)+"%.3f" % float(r[c]) for c in columns])
        except KeyError:
            row += ', '.join(["----" for c in columns])
    print(row)
