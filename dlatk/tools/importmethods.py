import csv
import json
import os, sys
import argparse
from MySQLdb import Warning
from MySQLdb.cursors import SSCursor

sys.path.append(os.path.dirname(os.path.realpath(__file__)).replace("/dlatk/tools",""))
from dlatk.mysqlmethods import mysqlMethods as mm

from warnings import filterwarnings
filterwarnings('ignore', category = Warning)

DEFAULT_DB = ''
DEFAULT_TABLE = ''
DEFAULT_CSV_FILE = ''
DEFAULT_JSON_FILE = ''

def appendCSVtoMySQL(csvFile, database, table, ignoreLines=0, dbCursor=None):
    if not dbCursor:
        dbConn, dbCursor, dictCursor = mm.dbConnect(database)
        dbCursor.execute("""SHOW TABLES LIKE '{table}'""".format(table=table))
        tables = [item[0] for item in dbCursor.fetchall()]
        if not tables:
            print("The table {table} does not exist in the database. Please use csvToMySQL or create the table.".format(table=table))
            sys.exit(1)
    disableSQL = """ALTER TABLE {table} DISABLE KEYS""".format(table=table)
    print(disableSQL)
    dbCursor.execute(disableSQL)
    print("""Importing data, reading {csvFile} file""".format(csvFile=csvFile))
    importSQL = """LOAD DATA LOCAL INFILE '{csvFile}' INTO TABLE {table} 
        FIELDS TERMINATED BY ',' ENCLOSED BY '"' 
        LINES TERMINATED BY '\r\n' IGNORE {ignoreLines} LINES""".format(csvFile=csvFile, table=table, ignoreLines=ignoreLines)
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

    createSQL = """CREATE TABLE {table} {colDesc}""".format(table=table, colDesc=columnDescription)
    print(createSQL)
    dbCursor.execute(createSQL)
    appendCSVtoMySQL(csvFile, database, table, ignoreLines, dbCursor)
    return

def jsonToMySQL(jsonFile, database, table, columnDescription):
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

def main():

    parser = argparse.ArgumentParser(description='Import / export methods for DLATK')
    parser.add_argument('-d', '--database', dest='db', default=DEFAULT_DB, help='MySQL database where tweets will be stored.')
    parser.add_argument('-t', '--table', dest='table', default=DEFAULT_TABLE, help='MySQL table name. If monthly tables then M_Y will be appended to end of this string. Default: %s' % (DEFAULT_TABLE))
    
    parser.add_argument('--csv_file', dest='csv_file', default=DEFAULT_CSV_FILE, help='Name and path to CSV file')
    parser.add_argument('--json_file', dest='json_file', default=DEFAULT_JSON_FILE, help='Name and path to JSON file')
    
    parser.add_argument('--column_description', dest='column_description', default=DEFAULT_CSV_FILE, help='Description of MySQL table.')
    parser.add_argument('--ignore_lines', type=int, dest='ignore_lines', default=0, help='Number of lines to ignore when uploading CSV.')

    parser.add_argument('--csv_to_mysql', action='store_true', dest='csv_to_mysql', default=False, help='Import CSV to MySQL')
    parser.add_argument('--append_csv_to_mysql', action='store_true', dest='append_csv_to_mysql', default=False, help='Append CSV to MySQL table')
    parser.add_argument('--json_to_mysql', action='store_true', dest='json_to_mysql', default=False, help='Import JSON to MySQL')
    parser.add_argument('--mysql_to_csv', action='store_true', dest='mysql_to_csv', default=False, help='Export MySQL table to CSV')

    args = parser.parse_args()

    if not args.db:
        print("You must choose a database -d")
        sys.exit(1)

    if not args.table:
        print("You must choose a table -t")
        sys.exit(1)

    if not (args.csv_to_mysql or args.json_to_mysql or args.mysql_to_csv or args.append_csv_to_mysql):
        print("You must choose some action: --csv_to_mysql, --json_to_mysql or --mysql_to_csv")
        sys.exit() 

    if args.csv_to_mysql:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        if not args.column_description:
            print("You must specify a column description --column_description")
            sys.exit(1)

        print("Importing {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
        csvToMySQL(args.csv_file, args.db, args.table, args.column_description, ignoreLines=args.ignore_lines)

    elif args.append_csv_to_mysql:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        print("Appending {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
        appendCSVtoMySQL(args.csv_file, args.db, args.table, ignoreLines=args.ignore_lines)

    elif args.json_to_mysql:
        print("--json_to_mysql is not implemented")
        jsonToMySQL(args.json_file, args.db, args.table, args.column_description)

    elif args.mysql_to_csv:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        print("Writing {db}.{table} to {csv}".format(db=args.db, table=args.table, csv=args.csv_file))
        mySQLToCSV(args.db, args.table, args.csv_file)

if __name__ == "__main__":
    main()
    sys.exit(0)