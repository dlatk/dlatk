import csv
import json
import os, sys
import argparse
try:
    from MySQLdb import Warning
    from MySQLdb.cursors import SSCursor
except:
    pass
import sqlite3
from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)).replace("/dlatk/tools",""))
from dlatk.mysqlmethods import mysqlMethods as mm
from dlatk.mysqlmethods import mysqlMethods as sm

from warnings import filterwarnings
filterwarnings('ignore', category = Warning)

DEFAULT_DB = ''
DEFAULT_TABLE = ''
DEFAULT_CSV_FILE = ''
DEFAULT_JSON_FILE = ''

# MySQL methods
def appendCSVtoMySQL(csvFile, database, table, ignoreLines=0, dbCursor=None):
    if not dbCursor:
        dbConn, dbCursor, dictCursor = mm.dbConnect(database)
        dbCursor.execute("""SHOW TABLES LIKE '{table}'""".format(table=table))
        tables = [item[0] for item in dbCursor.fetchall()]
        if not tables:
            print("The table {table} does not exist in the database. Please use csvToMySQL or create the table.".format(table=table))
            sys.exit(1)
    with open(csvFile, 'U') as f:
        f.readline()
        line_termination = f.newlines
    disableSQL = """ALTER TABLE {table} DISABLE KEYS""".format(table=table)
    print(disableSQL)
    dbCursor.execute(disableSQL)
    print("""Importing data, reading {csvFile} file""".format(csvFile=csvFile))
    importSQL = """LOAD DATA LOCAL INFILE '{csvFile}' INTO TABLE {table} 
        CHARACTER SET utf8mb4
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' 
        LINES TERMINATED BY '{lineTermination}' IGNORE {ignoreLines} LINES""".format(csvFile=csvFile, table=table, ignoreLines=ignoreLines, lineTermination=line_termination)
    dbCursor.execute(importSQL)
    enableSQL = """ALTER TABLE {table} ENABLE KEYS""".format(table=table)
    print(enableSQL)
    dbCursor.execute(enableSQL)
    return

def csvToMySQL(csvFile, database, table, columnDescription, ignoreLines=0):
    dbConn, dbCursor, dictCursor = mm.dbConnect(database)

    dbCursor.execute("""SHOW TABLES LIKE '{table}'""".format(table=table))
    tables = [item[0] for item in dbCursor.fetchall()]
    if tables:
        print("A table by that name already exists in the database. Please use appendCSVtoMySQL or choose a new name.")
        sys.exit(1)

    createSQL = """CREATE TABLE {table} {colDesc} CHARACTER SET utf8mb4""".format(table=table, colDesc=columnDescription)
    print(createSQL)
    dbCursor.execute(createSQL)
    appendCSVtoMySQL(csvFile, database, table, ignoreLines, dbCursor)
    return

def mySQLToCSV(database, table, csvFile, csvQuoting=csv.QUOTE_ALL):
    csvPath = os.path.dirname(os.path.abspath(csvFile))
    if not os.path.isdir(csvPath):
        print("Path {path} does not exist".format(path=csvPath))
        sys.exit(1)
    dbConn, dbCursor, dictCursor = mm.dbConnect(database)
    ssCursor = dbConn.cursor(SSCursor)
    ssCursor.execute("select * from {table}".format(table=table))
    header = [i[0] for i in ssCursor.description]
    with open(csvFile, "w") as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csvQuoting)
        csv_writer.writerow(header)
        csv_writer.writerows(ssCursor)
    return

# SQLite methods
def chunks(data, rows=10000):
    """ Divides the data into 10000 rows each """
    for i in range(0, len(data), rows):
        yield data[i:i+rows]
        
def csvToSQLite(csvFile, database, table, columnDescription, ignoreLines=0):
    dbConn, dbCursor = sm.dbConnect(database)
    dbCursor.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{table}'""".format(table=table))
    tables = [item[0] for item in dbCursor.fetchall()]
    if tables:
        print("A table by that name already exists in the database. Please use appendCSVtoSQLite or choose a new name.")
        sys.exit(1)
    createSQL = """CREATE TABLE {table} {colDesc}""".format(table=table, colDesc=columnDescription)
    print(createSQL)
    dbCursor.execute(createSQL)
    appendCSVtoSQLite(csvFile, database, table, ignoreLines, dbCursor, dbConn)
    return

def appendCSVtoSQLite(csvFile, database, table, ignoreLines=0, dbCursor=None, dbConn=None):
    if not dbCursor:
        dbConn, dbCursor = sm.dbConnect(database)
        dbCursor.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{table}'""".format(table=table))
        tables = [item[0] for item in dbCursor.fetchall()]
        if tables:
            print("A table by that name already exists in the database. Please use appendCSVtoSQLite or choose a new name.")
            sys.exit(1)
    print("""Importing data, reading {csvFile} file""".format(csvFile=csvFile))
    with open(csvFile, "r") as f:
        csvReader = csv.reader(f, delimiter=",")
        if ignoreLines > 0:
            for i in range(0,ignoreLines): 
                next(csvReader)
        csvData = list(csvReader)
        chunkData = chunks(csvData) 
        numColumns = None
        for chunk in chunkData:
            if not numColumns:
                numColumns = len(chunk[0])
                values_str = "(" + ",".join(["?"]*numColumns) + ")"
            dbCursor.executemany("""INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str), chunk)
            dbConn.commit()
    dbConn.close()
    return

def sqliteToCSV(database, table, csvFile, csvQuoting=csv.QUOTE_ALL):
    csvPath = os.path.dirname(os.path.abspath(csvFile))
    if not os.path.isdir(csvPath):
        print("Path {path} does not exist".format(path=csvPath))
        sys.exit(1)
    dbConn, dbCursor = sm.dbConnect(database)
    dbCursor.execute("select * from {table}".format(table=table))
    header = [i[0] for i in dbCursor.description]
    with open(csvFile, "w") as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csvQuoting)
        csv_writer.writerow(header)
        csv_writer.writerows(dbCursor)
    dbConn.close()
    return

# JSON methods
def jsonToMySQL(jsonFile, database, table, columnDescription):
    return


def main():

    parser = argparse.ArgumentParser(description='Import / export methods for DLATK')
    # MySQL flags
    parser.add_argument('-d', '--database', dest='db', default=DEFAULT_DB, help='MySQL database where tweets will be stored.')
    parser.add_argument('-t', '--table', dest='table', default=DEFAULT_TABLE, help='MySQL table name. If monthly tables then M_Y will be appended to end of this string. Default: %s' % (DEFAULT_TABLE))
    
    # file flags
    parser.add_argument('--csv_file', dest='csv_file', default=DEFAULT_CSV_FILE, help='Name and path to CSV file')
    parser.add_argument('--json_file', dest='json_file', default=DEFAULT_JSON_FILE, help='Name and path to JSON file')

    # action flags
    parser.add_argument('--csv_to_mysql', action='store_true', dest='csv_to_mysql', default=False, help='Import CSV to MySQL')
    parser.add_argument('--append_csv_to_mysql', action='store_true', dest='append_csv_to_mysql', default=False, help='Append CSV to MySQL table')
    parser.add_argument('--json_to_mysql', action='store_true', dest='json_to_mysql', default=False, help='Import JSON to MySQL')
    parser.add_argument('--mysql_to_csv', action='store_true', dest='mysql_to_csv', default=False, help='Export MySQL table to CSV')
    
    # other flags
    parser.add_argument('--column_description', dest='column_description', default=DEFAULT_CSV_FILE, help='Description of MySQL table.')
    parser.add_argument('--ignore_lines', type=int, dest='ignore_lines', default=0, help='Number of lines to ignore when uploading CSV.')

    args = parser.parse_args()

    # check that flags are properly set
    if not args.db:
        print("You must choose a database -d")
        sys.exit(1)

    if not args.table:
        print("You must choose a table -t")
        sys.exit(1)

    if not (args.csv_to_mysql or args.json_to_mysql or args.mysql_to_csv or args.append_csv_to_mysql):
        print("You must choose some action: --csv_to_mysql, --append_csv_to_mysql, --json_to_mysql or --mysql_to_csv")
        sys.exit() 

    ### perform actions
    # export actions
    if args.csv_to_mysql or args.append_csv_to_mysql:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        if not Path(args.csv_file).is_file():
            print("Your CSV file does not exist.")
            sys.exit(1)
        if args.csv_to_mysql:
            if not args.column_description:
                print("You must specify a column description --column_description")
                sys.exit(1)

            print("Importing {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
            csvToMySQL(args.csv_file, args.db, args.table, args.column_description, ignoreLines=args.ignore_lines)
        else:
            print("Appending {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
            appendCSVtoMySQL(args.csv_file, args.db, args.table, ignoreLines=args.ignore_lines)        

    elif args.json_to_mysql:
        print("--json_to_mysql is not implemented")
        jsonToMySQL(args.json_file, args.db, args.table, args.column_description)

    # import actions
    elif args.mysql_to_csv:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        print("Writing {db}.{table} to {csv}".format(db=args.db, table=args.table, csv=args.csv_file))
        mySQLToCSV(args.db, args.table, args.csv_file)

if __name__ == "__main__":
    main()
    sys.exit(0)
