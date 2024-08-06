import csv
import json
import os, sys
import argparse
try:
    from MySQLdb import Warning
    from MySQLdb.cursors import SSCursor
    import sqlite3
except:
    pass
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)).replace("/dlatk/tools",""))
from dlatk.mysqlmethods import mysqlMethods as mm
from dlatk.sqlitemethods import sqliteMethods as sm
from dlatk.database.dataEngine import DataEngine
from dlatk import dlaConstants as dlac

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

def csvToMySQL(csvFile, database, table,  mysql_config_file=dlac.MYSQL_CONFIG_FILE, encoding=dlac.DEF_ENCODING, use_unicode=dlac.DEF_UNICODE_SWITCH):

    data_engine = DataEngine(database, mysql_config_file, encoding, use_unicode, "mysql")
    (dbConn, dbCursor, dictCursor) = data_engine.connect()

    if not data_engine.tableExists(table):
        data_engine.dataEngine.csvToTable(csvFile, table)
    else:
        dlac.warn("Table alreadye exists")

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

def checkIfTableExists(table, dbCursor):
    dbCursor.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{table}'""".format(table=table))
    tables = [item[0] for item in dbCursor.fetchall()]
    if tables:
        print("A table called '{table}' already exists in the database.".format(table=table))
        sys.exit(1)
    else:
        return

# methods for importing ConvoKit data
# see documentation for format of ConvoKit data:
# https://convokit.cornell.edu/documentation/data_format.html
def doesTableExists(table, dbCursor):
    dbCursor.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{table}'""".format(table=table))
    tables = [item[0] for item in dbCursor.fetchall()]
    if tables:
        print("A table called '{table}' already exists in the database.".format(table=table))
        return False
    else:
        return True

def checkExactTypes(type):
    typeStr = "VARCHAR(255)"
    try:
        if type == 'string':
            typeStr = "VARCHAR(255)"
        elif type == 'Int64':
            typeStr = "INTEGER"
        elif type == 'Float64':
            typeStr = "FLOAT"
    except:
        pass
    return typeStr

def getColumnTypes(jsonFile, table, numColsToCheck = 5):
    idCol = "id"
    if table == "speakers":
        idCol = "speaker"
    elif table == "conversations":
        idCol = "conversation_id"
    elif table == "corpus":
        idCol = "meta_data"
    elif table == "index":
        idCol = "index"
    
    columns, types = [], []
    thisData = []

    with open(jsonFile) as f:
        data = json.load(f)
        i = 0
        for k, v in data.items():
            flatData = {idCol: k}
            for kk, vv in v.items():
                if kk == "meta":
                    metaData = v[kk]
                    for kkk, vvv in metaData.items():
                        if isinstance(vvv, list) or isinstance(vvv, dict):
                            vvv = str(vvv)
                        flatData[kkk] = vvv
                else:
                    if isinstance(vv, list) or isinstance(vv, dict):
                        vv = str(vv)
                    flatData[kk] = vv
            thisData.append(flatData)
            i += 1
            if i == numColsToCheck:
                break
        df = pd.DataFrame(thisData).convert_dtypes()
        for ii in zip(df.columns, df.dtypes):
            columns.append(ii[0])
            types.append(checkExactTypes(ii[1]))
    return columns, types

def getColumnTypesUtterances(jsonFile, numColsToCheck = 5):
    columns, types = [], []
    thisData = []
    with open(jsonFile) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            flatData = {}
            for k,v in data.items():
                if k == 'meta':
                    metaData = data[k]
                    for kk, vv in metaData.items():
                        if isinstance(vv, list) or isinstance(vv, dict):
                            vv = str(vv)
                        flatData[kk] = vv
                else:
                    if isinstance(v, list) or isinstance(v, dict):
                        v = str(v)
                    flatData[k] = v
            thisData.append(flatData)
            if i == numColsToCheck:
                break
        
        df = pd.DataFrame(thisData).convert_dtypes()
        for ii in zip(df.columns, df.dtypes):
            columns.append(ii[0])
            types.append(checkExactTypes(ii[1]))
        columns = [c if c != 'id' else 'message_id' for c in columns ]
        columns = [c if c != 'text' else 'message' for c in columns ]
    return columns, types

def createColDescription(columns, types, tableType=""):
    columnDescription = ""
    for column, type in zip(columns, types):
        if column == "id":
            columnDescription += "{column} {type} PRIMARY KEY, ".format(column=column, type=type)
        else:
            columnDescription += "{column} {type}, ".format(column=column, type=type)
    # if tableType == "utterances":
    #     columnDescription += """PRIMARY KEY (id)"""
    columnDescription = columnDescription[:-2]
    columnDescription = columnDescription.replace("-", "_")
    return columnDescription

def flattenUtterancesJSON(jsonData):
    dataToWrite = []
    for k,v in jsonData.items():
        if k == 'meta':
            metaData = jsonData[k]
            for kk, vv in metaData.items():
                if isinstance(vv, list) or isinstance(vv, dict):
                    vv = str(vv)
                dataToWrite.append(vv)
        else:
            if isinstance(v, list) or isinstance(v, dict):
                v = str(v)
            dataToWrite.append(v)
    return dataToWrite

def flattenJSON(jsonData):
    dataToWrite = []
    for k,v in jsonData.items():
        data = [k]
        for kk, vv in v.items():
            if kk == "meta":
                metaData = v[kk]
                for kkk, vvv in metaData.items():
                    if isinstance(vvv, list) or isinstance(vvv, dict):
                        vvv = str(vvv)
                    data.append(vvv)
            else:
                if isinstance(vv, list) or isinstance(vv, dict):
                    vv = str(vv)
                data.append(vv)
        dataToWrite.append(data)
    return dataToWrite

def importConvoKit(pathToCorpus):
    if not pathToCorpus.endswith("/"):
        pathToCorpus += "/"
    databaseName = Path(pathToCorpus).stem
    dbConn, dbCursor = sm.dbConnect(databaseName)

    tables = ["utterances", "speakers", "conversations", ] # "corpus", "index"

    for table in tables:
        jsonFile = pathToCorpus + table + ".json"
        if table == "utterances":
            jsonFile += "l"
        if os.path.isfile(jsonFile) and doesTableExists(table, dbCursor):
            if table == "utterances":
                columns, types = getColumnTypesUtterances(jsonFile)
            else:
                columns, types = getColumnTypes(jsonFile, table)
            columnDescription = createColDescription(columns, types, table)
            createSQL = """CREATE TABLE {table} ({colDesc});""".format(table=table, colDesc=columnDescription)
            print(createSQL)
            dbCursor.execute(createSQL)

            print("""Importing data, reading {csvFile} file""".format(csvFile=jsonFile))
            if table == "utterances":
                dataToWrite = []
                numColumns = None
                with open(jsonFile) as f:
                    for i, line in enumerate(f, 1):
                        data = json.loads(line)
                        data = flattenUtterancesJSON(data)
                        if not numColumns:
                            numColumns = len(data)
                            values_str = "(" + ",".join(["?"]*numColumns) + ")"
                        dataToWrite.append(data)
                        if i % 10000 == 0:
                            print("\tWrote {i} lines".format(i=i))
                            dbCursor.executemany("""INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str), dataToWrite)
                            dbConn.commit()
                            dataToWrite = []
                    if len(dataToWrite) > 0:
                        print("\tWrote {i} lines".format(i=i))
                        dbCursor.executemany("""INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str), dataToWrite)
                        dbConn.commit()
                        dataToWrite = []

            else:
                with open(jsonFile) as f:
                    data = json.load(f)
                data = flattenJSON(data)
                chunkData = chunks(data)
                numColumns = None
                for chunk in chunkData:
                    if not numColumns:
                        numColumns = len(chunk[0])
                        values_str = "(" + ",".join(["?"]*numColumns) + ")"
                    dbCursor.executemany("""INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str), chunk)
                    dbConn.commit()
            indexSQL = []
            if table == "utterances":
                indexSQL = ["""CREATE UNIQUE INDEX ut_id_idx ON utterances (message_id);""",
                            """CREATE INDEX ut_speaker_idx ON utterances (speaker);""",
                            """CREATE INDEX ut_conversation_id_idx ON utterances (conversation_id);""",
                            ]
            elif table == "speakers":
                indexSQL = ["""CREATE UNIQUE INDEX sp_speaker_idx ON speakers (speaker);"""]
            elif table == "conversations":
                indexSQL = ["""CREATE UNIQUE INDEX co_conversation_id_idx ON conversations (conversation_id);"""]
            if indexSQL:
                for isql in indexSQL:
                    print(isql)
                    dbCursor.execute(isql)
        else:
            print("The file {file} does not exist, skipping.".format(file=pathToCorpus + table + ".jsonl"))
            pass
    dbConn.close()
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
    parser.add_argument('--csv_to_sqlite', action='store_true', dest='csv_to_sqlite', default=False, help='Import CSV to SQLite')
    
    # other flags
    parser.add_argument('--column_description', dest='column_description', default=DEFAULT_CSV_FILE, help='Description of MySQL table.')
    parser.add_argument('--ignore_lines', type=int, dest='ignore_lines', default=0, help='Number of lines to ignore when uploading CSV.')

    parser.add_argument('--add_convokit', dest='convokit_data', default="", help='Path to ConvoKit formatted data.')
    

    args = parser.parse_args()

    # check that flags are properly set
    if not args.db and not args.convokit_data:
        print("You must choose a database -d")
        sys.exit(1)

    if not args.table and not args.convokit_data:
        print("You must choose a table -t")
        sys.exit(1)

    if not (args.csv_to_mysql or args.json_to_mysql or args.mysql_to_csv or args.append_csv_to_mysql or args.csv_to_sqlite or args.convokit_data):
        print("You must choose some action: --csv_to_mysql, --append_csv_to_mysql, --json_to_mysql or --mysql_to_csv or --csv_to_sqlite or --add_convokit")
        sys.exit() 

    ### perform actions
    # import
    if args.csv_to_mysql or args.append_csv_to_mysql or args.csv_to_sqlite:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        if not Path(args.csv_file).is_file():
            print("Your CSV file does not exist.")
            sys.exit(1)
        if args.csv_to_mysql:
            print("Importing {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
            csvToMySQL(args.csv_file, args.db, args.table)
        elif args.csv_to_sqlite:
            if not args.column_description:
                print("You must specify a column description --column_description")
                sys.exit(1)
            print("Importing {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
            csvToSQLite(args.csv_file, args.db, args.table, args.column_description, ignoreLines=args.ignore_lines)
        else:
            print("Appending {csv} to {db}.{table}".format(db=args.db, table=args.table, csv=args.csv_file))
            appendCSVtoMySQL(args.csv_file, args.db, args.table, ignoreLines=args.ignore_lines)        

    elif args.json_to_mysql:
        print("--json_to_mysql is not implemented")
        jsonToMySQL(args.json_file, args.db, args.table, args.column_description)
    elif args.convokit_data:
        importConvoKit(args.convokit_data)

    # export actions
    elif args.mysql_to_csv:
        if not args.csv_file:
            print("You must specify a csv file --csv_file")
            sys.exit(1)
        print("Writing {db}.{table} to {csv}".format(db=args.db, table=args.table, csv=args.csv_file))
        mySQLToCSV(args.db, args.table, args.csv_file)

if __name__ == "__main__":
    main()
    sys.exit(0)
