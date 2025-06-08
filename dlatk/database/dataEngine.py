from ..mysqlmethods import mysqlMethods as mm
from ..sqlitemethods import sqliteMethods as sm
from .. import dlaConstants as dlac
from collections import defaultdict
from subprocess import check_output
import random
import sys
import os
import csv
import ast
import json

class DataEngine(object):
    """
    Class for connecting with the database engine (based on the type of data engine being used) and executing queries.

    Parameters
    -------------
    corpdb: str
        Corpus Database name.
    mysql_config_file : str
        Location of MySQL configuration file
    encoding: str
        MySQL encoding
    db_type: str
        Type of the database being used (mysql, sqlite).
    
    """
    def __init__(self, corpdb=dlac.DEF_CORPDB, mysql_config_file=dlac.MYSQL_CONFIG_FILE, encoding=dlac.DEF_ENCODING,use_unicode=dlac.DEF_UNICODE_SWITCH, db_type=dlac.DB_TYPE):
        self.encoding = encoding
        self.corpdb = corpdb
        self.mysql_config_file = mysql_config_file
        self.use_unicode = use_unicode
        self.db_type = db_type
        self.dataEngine = None

    def connect(self):
        """
        Establishes connection with the database engine
    
        Returns
        -------------
        Database connection objects
        """
        if self.db_type == "mysql":
            self.dataEngine = MySqlDataEngine(self.corpdb, self.mysql_config_file, self.encoding)
        if self.db_type == "sqlite":
            self.dataEngine = SqliteDataEngine(self.corpdb)
        return self.dataEngine.get_db_connection()

    def disable_table_keys(self, featureTableName):
        """
        Disable keys: good before doing a lot of inserts.

        Parameters
        ------------
        featureTableName: str
            Name of the feature table
        """
        self.dataEngine.disable_table_keys(featureTableName)

    def enable_table_keys(self, featureTableName):
        """
        Enables the keys, for use after inserting (and with keys disabled)

        Parameters
        ------------
        featureTableName: str
            Name of the feature table
        """
        self.dataEngine.enable_table_keys(featureTableName)

    def execute_get_list(self, usql):
        """
        Executes the given select query

        Parameters
        ------------
        usql: str
            SELECT sql statement to execute        

        Returns
        ------------
        Results as list of lists

        """
        return self.dataEngine.execute_get_list(usql)


    def execute_get_SSCursor(self, usql):
        """
        Executes the given select query

        Parameters
        ------------
        usql: str
            SELECT sql statement to execute        

        Returns
        ------------
        Results as list of lists

        """
        return self.dataEngine.execute_get_SSCursor(usql)

    def execute_write_many(self, usql, insert_rows):
        """
        Executes the given insert query
        
        Parameters
        ------------
        usql: str
            Insert statement
        insert_rows: list
            List of rows to insert into table 
        
        """
        #print(usql) #DEBUG
        self.dataEngine.execute_write_many(usql, insert_rows)

    def execute(self, sql):
        """
        Executes a given query
        
        Parameters
        ------------
        sql: str

        Returns
        ------------
        True, if the query execution is successful
        """
        return self.dataEngine.execute(sql)

    def standardizeTable(self, table, collate, engine, charset, use_unicode):
        """
        Parameters
        ------------
        table: str
            Name of the table
        collate: str
            Collation
        engine: str
            Database engine (mysql engine)
        charset: str
            Character set encoding
        use_unicode: bool
            Use unicode if True
        
        Returns
        ------------
        True, if the query execution is successful
        """
        return self.dataEngine.standardizeTable(table, collate, engine, charset, use_unicode)

    def tableExists(self, table_name):
        """
        Checks whether a table exists
        
        Parameters
        ------------
        table_name: str

        Returns
        ------------
        True or False
        """
        return self.dataEngine.tableExists(table_name)

    def primaryKeyExists(self, table_name, column_name):
        """
        Checks whether a primary key exists in table_name on column_name

        Parameters
        ------------
        table_name: str
        
        column_name: str

        Returns
        ------------
        True or False
        """
        return self.dataEngine.primaryKeyExists(table_name, column_name)

    def indexExists(self, table_name, column_name):
        """
        Checks whether an index (which is not a primary key) exists
        
        Parameters
        ------------
        table_name: str
        
        column_name: str

        Returns
        ------------
        True or False
        """
        return self.dataEngine.indexExists(table_name, column_name)
    
    def getTables(self, like, feat_table=False):
        """
        Returns a list of available tables.
 
        Parameters
        ----------
        like: boolean
        Filter tables with sql-style call.

        feat_table: str, optional
        Indicator for listing feature tables

        Returns
        -------
        list
        """
        return self.dataEngine.getTables(like, feat_table)

    def describeTable(self, table_name):
        """
         Describe the column names and column types of a table.

        Parameters
        ----------
        table_name: str
        Name of table to describe

        Returns
        -------
        list of lists 
        """
        return self.dataEngine.describeTable(table_name)    

    def getRandomFunc(self, random_seed):
        """
        Gives the right "random" function based on data engine used.

        Parameters
        ----------
        random_seed: float
        Seed value used to produce the random number

        Returns
        -------
        str
    
        """
        return self.dataEngine.getRandomFunc(random_seed)

    def viewTable(self, table_name):
        """
        Returns the first 5 rows of table (to inspect).

        Parameters
        ----------
        table_name : :obj:`str`
        Name of table to describe

        Returns
        -------
        list of lists
        """
        return self.dataEngine.viewTable(table_name)

    def getTableColumnNameTypes(self, table_name):
        """
        return a dict of column names mapped to types

        Parameters
        -------------
        table_name: str

        Returns
        -------------
        Dict
        """
        return self.dataEngine.getTableColumnNameTypes(table_name)
    
    def checkExactTypes(self, type):
        typeStr = "TEXT"
        try:
            if type == 'string':
                typeStr = "TEXT"
            elif type == 'Int64':
                typeStr = "INTEGER"
            elif type == 'Float64':
                typeStr = "FLOAT"
        except:
            pass
        return typeStr
    
    def flattenUtterancesJSON(self, jsonData, numRowsToKeep = -1):
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

    def flattenJSON(self, jsonData, table, numRowsToKeep = -1):
        columnNames = ["id"]
        if table == "speakers":
            columnNames = ["speaker"]
        elif table == "conversations":
            columnNames = ["conversation_id"]
        elif table == "corpus":
            columnNames = ["meta_data"]
        elif table == "index":
            columnNames = ["index"]
        
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
                        if len(dataToWrite) == 0: columnNames.append(kkk)
                else:
                    if isinstance(vv, list) or isinstance(vv, dict):
                        vv = str(vv)
                    data.append(vv)
                    if len(dataToWrite) == 0: columnNames.append(kk)
            dataToWrite.append(data)
            if numRowsToKeep > 0 and len(dataToWrite) == numRowsToKeep:
                return columnNames, dataToWrite
        return dataToWrite

    def getSampleUtterances(self, jsonFile, numRowsToCheck = 1000):
        columnNames, dataToWrite = [], []
        with open(jsonFile) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                thisData = []
                for k,v in data.items():
                    if k == 'meta':
                        metaData = data[k]
                        for kk, vv in metaData.items():
                            if isinstance(vv, list) or isinstance(vv, dict):
                                vv = str(vv)
                            thisData.append(vv)
                            if len(dataToWrite) == 0: columnNames.append(kk)
                    else:
                        if isinstance(v, list) or isinstance(v, dict):
                            v = str(v)
                        thisData.append(v)
                        if len(dataToWrite) == 0: columnNames.append(k)
                dataToWrite.append(thisData)
                if i == numRowsToCheck:
                    break
        columnNames[columnNames.index("id")] = "message_id"
        return columnNames, dataToWrite
    
    def createColDescription(self, columns, types, tableType=""):
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
    
    def importConvoKit(self, pathToCorpus):
        if not pathToCorpus.endswith("/"):
            pathToCorpus += "/"
        
        tables = ["utterances", "speakers", "conversations", ] # "corpus", "index"

        for table in tables:
            jsonFile = pathToCorpus + table + ".json"
            if table == "utterances":
                jsonFile += "l"
            if os.path.isfile(jsonFile) and not self.tableExists(table):
                print("""Importing data, reading {csvFile} file""".format(csvFile=jsonFile))
                if table == "utterances":
                    columnNames, sample = self.getSampleUtterances(jsonFile)
                    columnDescription, numColumns = self.get_column_description(columnNames, sample)
                    columnDescription = self.createColDescription([c[0] for c in columnDescription], [c[1] for c in columnDescription], table)
                    createSQL = """CREATE TABLE {table} ({colDesc});""".format(table=table, colDesc=columnDescription)
                    print(createSQL)
                    self.execute(createSQL)

                    dataToWrite = []
                    numColumns = None
                    with open(jsonFile) as f:
                        for i, line in enumerate(f, 1):
                            data = json.loads(line)

                            data = self.flattenUtterancesJSON(data)
                            if not numColumns:
                                numColumns = len(data)
                                placeholder = "%s" if self.db_type == "mysql" else "?"
                                values_str = "(" + ",".join([placeholder]*numColumns) + ")"
                            dataToWrite.append(tuple(data))
                            if i % 10000 == 0:
                                insertQuery = """INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str)
                                self.execute_write_many(insertQuery, dataToWrite)
                                print("\tWrote {i} lines".format(i=i))
                                dataToWrite = []
                        if len(dataToWrite) > 0:
                            insertQuery = """INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str)
                            self.execute_write_many(insertQuery, dataToWrite)
                            print("\tWrote {i} lines".format(i=i))
                            dataToWrite = []
                else:
                    with open(jsonFile) as f:
                        data = json.load(f)
                    
                    columnNames, sample = self.flattenJSON(data, table, numRowsToKeep=40000)
                    columnDescription, numColumns = self.get_column_description(columnNames, sample)
                    columnDescription = self.createColDescription([c[0] for c in columnDescription], [c[1] for c in columnDescription], table)
                    createSQL = """CREATE TABLE {table} ({colDesc});""".format(table=table, colDesc=columnDescription)
                    print(createSQL)
                    self.execute(createSQL)

                    data = self.flattenJSON(data, table)
                    chunkData = self.chunks(data)
                    numColumns = None
                    for chunk in chunkData:
                        if not numColumns:
                            numColumns = len(chunk[0])
                            placeholder = "%s" if self.db_type == "mysql" else "?"
                            values_str = "(" + ",".join([placeholder]*numColumns) + ")"
                        insertQuery = """INSERT INTO {table} VALUES {values}""".format(table=table, values=values_str)
                        self.execute_write_many(insertQuery, chunk)
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
                        self.execute(isql)
            else:
                print("The file {file} does not exist or the table already exists in the database, skipping.".format(file=pathToCorpus + table + ".jsonl"))
                pass
        return 
    
    def read_csv_sample(self, csv_file):
        #read a random sample of size 1000 from the CSV.
        sample_size = 1000
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            sample = []
            for idx in range(sample_size):
                try:
                    row = next(reader)
                    sample.append(row)
                except StopIteration:
                    break
        return header, sample
    
    def chunks(self, data, rows=10000):
        """ Divides the data into 10000 rows each """
        for i in range(0, len(data), rows):
            yield data[i:i+rows]
    
    def get_column_description(self, column_names, sample):
        """
        Infers the column datatypes from a CSV and returns the SQL column description.
        
        Parameters
        ------------
        csv_file: str
        """

        def _eval(x):
            try:
                return ast.literal_eval(x)
            except:
                return x

        def _next_power_of_2(x):  
            return 1 if x == 0 else 2**(x+2).bit_length()


        #infer the column types from the sample.
        max_vc_length = 100
        num_columns = len(column_names)
        column_description = []    
        for cid in range(num_columns):

            length, is_str, column_type = 0, False, "TEXT"
            for index, row in enumerate(sample):
                column_value = _eval(row[cid])
                if column_value == '': continue #column type isn't infered until you see the first non-null value
                if (not is_str) and (isinstance(column_value, int)):
                    column_type = "INT"
                elif (not is_str) and (isinstance(column_value, float)):
                    column_type = "DOUBLE"
                else:
                    is_str = True
                    column_value = str(column_value)
                    length = max(len(column_value), length)
                    if length <= max_vc_length:
                        #column_type = "VARCHAR({})".format(length)
                        column_type = "VARCHAR({})".format(_next_power_of_2(length)-1)
                    else:
                        column_type = "LONGTEXT"
                        break

            column_description.append((column_names[cid], column_type))

        return column_description, num_columns

    def csvToTable(self, csv_file, table_name):
        """
        Loads a CSV file as a SQLite table to the database
        
        Parameters
        ------------
        csv_file: str

        table_name: str
        """
        header, sample = self.read_csv_sample(csv_file)
        column_description, num_columns = self.get_column_description(header, sample)

        createSQL = '(' + ', '.join(["{} {}".format(cname, ctype) for cname, ctype in column_description]) + ");"
        createSQL = "CREATE TABLE {} {}".format(table_name, createSQL)
        self.execute(createSQL)

        print("Importing data, reading {} file".format(csv_file))

        placeholder = "%s" if self.db_type == "mysql" else "?"
        values_str = "(" + ",".join([placeholder] * num_columns) + ")"
        insertQuery = "INSERT INTO {} VALUES {}".format(table_name, values_str)

        type_map = {"INT": int, "DOUBLE": float}
        with open(csv_file, 'r', errors="surrogateescape") as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            chunk = []
            for index, row in enumerate(reader):

                if (index % 10000 == 0 and index > 0):
                    print("Reading {} rows into the table...".format(len(chunk)))
                    self.execute_write_many(insertQuery, chunk)
                    chunk = []

                #enforcing strict column type
                cleaned_row = []
                for cindex, cvalue in enumerate(row):
                    try:
                        cvalue = type_map.get(column_description[cindex][1], str)(cvalue)
                        cleaned_row.append(cvalue)
                    except ValueError as e:
                        cleaned_row.append(None)
                chunk.append(cleaned_row)

            print("Reading remaining {} rows into the table...".format(len(chunk)))
            self.execute_write_many(insertQuery, chunk)

    def tableToCSV(self, table_name, csv_file, quoting=csv.QUOTE_ALL):
        """
        Dumps the SQLite table into a CSV file.
        
        Parameters
        ------------
        table_name: str

        csv_file: str

        quoting: [csv.QUOTE_ALL | csv.QUOTE_MINIMAL | csv.QUOTE_NONNUMERIC | csv.QUOTE_NONE]
        """

        path = os.path.dirname(os.path.abspath(csv_file))
        if not os.path.isdir(path):
            print("Path {path} does not exist".format(path=path))
            sys.exit(1)
        
        selectQuery = "SELECT * FROM {}".format(table_name)
        self.dbCursor.execute(selectQuery)
        header = [i[0] for i in self.dbCursor.description]
        with open(csv_file, 'w') as f:
            csv_writer = csv.writer(f, quoting=quoting)
            csv_writer.writerow(header)
            csv_writer.writerows(self.dbCursor)

        return

    def exportConvoKit(self, outputPath, feature_tables=None):
        """
        Exports data from the database to ConvoKit-compatible JSON/JSONL files,
        adding extracted features to the 'meta' field of the appropriate table.

        Parameters:
            outputPath (str): Directory where the JSON files will be saved.
        """

        if not outputPath.endswith("/"):
            outputPath += "/"

        # Ensure the output directory exists
        os.makedirs(outputPath, exist_ok=True)

        # Define main tables to export
        main_tables = ["utterances", "speakers", "conversations"]

        # Initialize dictionaries to hold records for each table
        records_dicts = {table: {} for table in main_tables}

        # First, fetch and store all records for each main table
        for main_table in main_tables:
            jsonFile = os.path.join(outputPath, main_table + (".jsonl" if main_table == "utterances" else ".json"))

            # Check if the main table exists
            if not self.tableExists(main_table):
                print("Table '{}' does not exist in the database. Skipping export for this table.".format(main_table))
                continue

            print("Exporting table '{}' to '{}'...".format(main_table, jsonFile))
            # Fetch all records from the main table
            columns, rows = self.fetchAllRecords(main_table)

            if main_table == "utterances":
                records_dicts[main_table] = [dict(zip(columns, row)) for row in rows]
            else:
                key_field = "speaker" if main_table == "speakers" else "conversation_id"
                for row in rows:
                    record = dict(zip(columns, row))
                    key = record.get(key_field)
                    if key:
                        records_dicts[main_table][key] = record
                    else:
                        print("Record missing key field '{}': {}".format(key_field, record))

        # Now, handle feature extraction and merging based on group_id_col
        if feature_tables is None:
            feature_tables = self.getFeatureTables("utterances")
        else:
            feature_tables = feature_tables.split(',')

        # Fetch all features from each feature table and organize them
        features_by_group = self.fetchFeatures(feature_tables, "utterances")

        # Identify all group_id columns used in feature tables
        group_id_columns = self.getGroupIdColumns("utterances")
        if not group_id_columns:
            print("No group_id columns found for table 'utterances'. Skipping feature merging.")
        else:
            # Process each group_id_col separately
            for group_id_col in group_id_columns:
                print("Processing features based on group_id column '{}'.".format(group_id_col))

                # Determine the target table based on group_id_col
                if group_id_col == "speaker":
                    target_table = "speakers"
                    is_jsonl = False
                elif group_id_col == "conversation_id":
                    target_table = "conversations"
                    is_jsonl = False
                elif group_id_col == "message_id":
                    target_table = "utterances"
                    is_jsonl = True
                else:
                    print("Unknown group_id_col '{}'. Skipping.".format(group_id_col))
                    continue

                print("Adding features to '{}' based on '{}'.".format(target_table, group_id_col))

                # Open the appropriate JSON file for writing/updating
                jsonFile = os.path.join(outputPath, target_table + (".jsonl" if is_jsonl else ".json"))

                if is_jsonl:
                    # For utterances, write each record as a JSONL line
                    with open(jsonFile, 'w', encoding='utf-8') as f:
                        for record in records_dicts[target_table]:
                            group_id_val = record.get(group_id_col)
                            if group_id_val and group_id_val in features_by_group:
                                features = features_by_group[group_id_val]
                                for primary_key, feats in features.items():
                                    feature_key = primary_key.split("_")[0]
                                    if feature_key not in record:
                                        record[feature_key] = feats
                                    else:
                                        if isinstance(record[feature_key], dict) and isinstance(feats, dict):
                                            record[feature_key].update(feats)
                                        else:
                                            raise TypeError("Conflict at key '{}': both must be dictionaries to merge.".format(feature_key))

                            f.write(json.dumps(record) + "\n")
                    print("Exported '{}' to '{}' with features added to 'meta'.".format(target_table, jsonFile))
                else:
                    # For speakers and conversations, update the dictionary and write once
                    for key, record in records_dicts[target_table].items():
                        if key in features_by_group:
                            features = features_by_group[key]
                            for primary_key, feats in features.items():
                                feature_key = primary_key.split("_")[0]
                                if feature_key not in record:
                                    record[feature_key] = feats
                                else:
                                    if isinstance(record[feature_key], dict) and isinstance(feats, dict):
                                        record[feature_key].update(feats)
                                    else:
                                        raise TypeError("Conflict at key '{}': both must be dictionaries to merge.".format(feature_key))

                    # Write the updated records to JSON file
                    with open(jsonFile, 'w', encoding='utf-8') as f:
                        json.dump(records_dicts[target_table], f, indent=4)
                    print("Exported '{}' to '{}' with features added to 'meta'.".format(target_table, jsonFile))

        # Finally, export the records that are not related to features
        for main_table in main_tables:
            if main_table == "utterances":
                jsonFile = os.path.join(outputPath, main_table + ".jsonl")
                with open(jsonFile, 'w', encoding='utf-8') as f:
                    for record in records_dicts[main_table]:
                        f.write(json.dumps(record) + "\n")
                print("Final export of '{}' completed.".format(main_table))
            else:
                jsonFile = os.path.join(outputPath, main_table + ".json")
                with open(jsonFile, 'w', encoding='utf-8') as f:
                    json.dump(records_dicts[main_table], f, indent=4)
                print("Final export of '{}' completed.".format(main_table))

        print("Export completed.")
        return

    def fetchAllRecords(self, table_name):
        """
        Fetches all records from a specified table.

        Parameters:
            table_name (str): Name of the table to fetch.

        Returns:
            tuple: (columns, rows)
                - columns (list): List of column names.
                - rows (list of tuples): List of records.
        """

        describe_output = self.describeTable(table_name)
        if self.db_type == 'mysql':
            columns = [col[0] for col in describe_output]
        elif self.db_type == 'sqlite':
            columns = [col[1] for col in describe_output]

        select_query = "SELECT * FROM {} LIMIT 2;".format(table_name)
        rows = self.execute_get_list(select_query)
        return columns, rows

    def getFeatureTables(self, main_table):
        """
        Retrieves all feature tables related to the main_table based on naming convention.

        Parameters:
            main_table (str): The main table name (e.g., 'utterances').

        Returns:
            list: List of feature table names.
        """
        like_pattern = "feat$%${}$%".format(main_table)
        if self.db_type == 'mysql':
            fetched_rows = self.execute_get_list("""
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE '{}';
                """.format(like_pattern))
        elif self.db_type == 'sqlite':
            fetched_rows = self.execute_get_list("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'feat$%$%$%';
                """)
        else:
            raise ValueError("Unsupported database type.")

        tables = [row[0] for row in fetched_rows]

        feature_tables = [
            table for table in tables
            if len(table.split('$')) > 2 and table.split('$')[2] == main_table and not table.startswith('feat$meta_')
        ]
        return feature_tables

    def parseFeatureTableName(self, feature_table, main_table):
        """
        Parses the feature table name to extract feature_type and group_id.

        Parameters:
            feature_table (str): The feature table name (e.g., 'feat$1gram$utterances$speaker').
            main_table (str): The main table name (e.g., 'utterances').

        Returns:
            tuple: (feature_type, group_id)
        """
        parts = feature_table.split('$')
        if len(parts) != 4 or parts[2] != main_table:
            raise ValueError("Invalid feature table name format: {}".format(feature_table))
        feature_type = parts[1]
        group_id = parts[3]
        return feature_type, group_id

    def getGroupIdColumns(self, main_table):
        """
        Identifies the group_id columns in the main_table.

        Parameters:
            main_table (str): The main table name (e.g., 'utterances').

        Returns:
            list: List of group_id column names.
        """
        feature_tables = self.getFeatureTables(main_table)
        group_ids = set()
        for ft in feature_tables:
            try:
                _, group_id = self.parseFeatureTableName(ft, main_table)
                group_ids.add(group_id)
            except ValueError as e:
                print(e)
        return list(group_ids)

    def fetchFeatures(self, feature_tables, main_table):
        """
        Fetches and organizes feature data from feature tables.

        Parameters:
            feature_tables (list): List of feature table names.
            main_table (str): The main table name.

        Returns:
            dict: {group_id_val: {feature_key: {feat: {value, group_norm}}}}
        """
        features_by_group = defaultdict(lambda: defaultdict(dict))

        for feature_table in feature_tables:
            try:
                feature_type, group_id = self.parseFeatureTableName(feature_table, main_table)
            except ValueError as e:
                print(e)
                continue

            # Fetch features: group_id, feat, value, group_norm
            select_query = "SELECT group_id, feat, value, group_norm FROM {};".format(feature_table)
            feature_rows = self.execute_get_list(select_query)

            for group_id_val, feat, value, group_norm in feature_rows:
                feature_key = "{}_{}".format(feature_type, group_id)
                features_by_group[group_id_val][feature_key][feat] = {
                    "value": value,
                    "group_norm": group_norm
                }

        return features_by_group

class MySqlDataEngine(DataEngine):
    """
    Class for interacting with the MYSQL database engine.
    Parameters
    ------------
    corpdb: str
        Corpus database name.
    mysql_config_file : str
        Location of MySQL configuration file
    encoding: str
        MYSQL encoding
    """

    def __init__(self, corpdb, mysql_config_file, encoding):
        super().__init__(corpdb, mysql_config_file=mysql_config_file)
        self.mysql_config_file = mysql_config_file
        (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(corpdb, charset=encoding, mysql_config_file=mysql_config_file)

    def get_db_connection(self):
        """
        Returns
        ------------
        Database connection objects
        """
        return self.dbConn, self.dbCursor, self.dictCursor

    def getTables(self, like, feat_table):
        """
        Returns a list of available tables.
 
        Parameters
        ----------
        like: boolean
        Filter tables with sql-style call.

        feat_table: str, optional
        Indicator for listing feature tables

        Returns
        -------
        list
        """
        if feat_table:
            sql = "SHOW TABLES FROM {} LIKE '{}'".format(self.corpdb, like)
        else:
            sql = "SHOW TABLES FROM {} where Tables_in_{} NOT LIKE 'feat%%' ".format(self.corpdb, self.corpdb)
            if isinstance(like, str): sql += " AND Tables_in_{} like '{}'".format(self.corpdb, like)

        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def describeTable(self, table_name):
        """
         Describe the column names and column types of a table.

        Parameters
        ----------
        table_name: str
        Name of table to describe
     
        Returns
        -------
        list of lists 
        """

        sql = "DESCRIBE %s""" % (table_name)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def execute_get_list(self, usql):
        """
        Executes the given select query

        Parameters
        ------------
        usql: str
            SELECT sql statement to execute        

        Returns
        ------------
        Results as list of lists

        """
        return mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def execute_get_SSCursor(self, usql):
        """
        Executes the given select query

        Parameters
        ------------
        usql: str
            SELECT sql statement to execute        

        Returns
        ------------
        Results as list of lists

        """
        return mm.executeGetSSCursor(self.corpdb, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def disable_table_keys(self, featureTableName):
        """
        Disable keys: good before doing a lot of inserts.
        """
        mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def enable_table_keys(self, featureTableName):
        """
        Enables the keys, for use after inserting (and with keys disabled)

        Parameters
        ------------
        featureTableName: str
            Name of the feature table
        """
        mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def execute_write_many(self, wsql, insert_rows):
        """
        Executes the given insert query
        
        Parameters
        ------------
        usql: string
            Insert statement
        insert_rows: list
            List of rows to insert into table 
        
        """
        mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, insert_rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def execute(self, sql):
        """
        Executes a given query

        Parameters
        ------------
        sql: str
            
        Returns
        ------------
        True or False depending on the success of query execution
        """
        return mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def standardizeTable(self, table, collate, engine, charset, use_unicode):
        """
        Sets character set, collate and engine

        Parameers
        ------------
        table: str
            Name of the table
        collate: str
            Collation
        engine: str
            Database engine (mysql engine)
        charset: str
            Character set encoding
        use_unicode: bool
            Use unicode if True

        Returns
        ------------
        True if the query execution is successful 
        """
        return mm.standardizeTable(self.corpdb, self.dbCursor, table, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def tableExists(self, table_name):
        """
        Checks whether a table exists
        
        Parameters
        ------------
        table_name: str

        Returns
        ------------
        True or False
        """
        return mm.tableExists(self.corpdb, self.dbCursor, table_name, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

    def primaryKeyExists(self, table_name, column_name):
        """
        Checks whether a primary key exists in table_name on column_name

        Parameters
        ------------
        table_name: str
        
        column_name: str

        Returns
        ------------
        True or False
        """
        return mm.primaryKeyExists(self.corpdb, self.dbCursor, table_name, column_name, mysql_config_file=self.mysql_config_file)

    def indexExists(self, table_name, column_name):
        """
        Checks whether an index (which is not a primary key) exists
        
        Parameters
        ------------
        table_name: str
        
        column_name: str

        Returns
        ------------
        True or False
        """
        return mm.indexExists(self.corpdb, self.dbCursor, table_name, column_name, mysql_config_file=self.mysql_config_file)

    def getTableColumnNameTypes(self, table_name):
        """
        return a dict of column names mapped to types

        Parameters
        -------------
        table_name: str

        Returns
        -------------
        Dict
        """
        return mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, table_name, mysql_config_file=self.mysql_config_file)

    def viewTable(self, table_name):
        """
        Returns the first 5 rows of table (to inspect).

        Parameters
        ----------
        table_name : :obj:`str`
        Name of table to describe

        Returns
        -------
        list of lists
        """
        col_sql = "SELECT column_name FROM information_schema.columns WHERE table_schema = '{}' and table_name='{}'".format(self.corpdb, table_name)
        col_names = [col[0] for col in mm.executeGetList(self.corpdb, self.dbCursor, col_sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)]

        sql = """SELECT * FROM %s LIMIT 10""" % (table_name)
        return [col_names] + list(mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))

    def getRandomFunc(self, random_seed):
        """
        Gives the right "random" function based on data engine used.

        Parameters
        ----------
        random_seed: float
        Seed value used to produce the random number

        Returns
        -------
        str 
        """
        return "RAND({})".format(random_seed)

class SqliteDataEngine(DataEngine):
    def __init__(self, corpdb):
        super().__init__(corpdb, db_type="sqlite")
        (self.dbConn, self.dbCursor) = sm.dbConnect(corpdb)

    def get_db_connection(self):
        """
        Returns
        ------------
        Database connection objects
        """
        return self.dbConn, self.dbCursor, None

    def getTables(self, like, feat_table):
        """
        Returns a list of available tables.
 
        Parameters
        ----------
        like: boolean
        Filter tables with sql-style call.

        feat_table: str, optional
        Indicator for listing feature tables
 
        Returns
        -------
        list
        """
        if feat_table:
            sql = "SELECT name FROM sqlite_master WHERE (type='table') AND (name LIKE '{}')".format(like)
        else:
            sql = "SELECT name FROM sqlite_master WHERE (type='table') AND (name NOT LIKE 'feat%%') "
            if isinstance(like, str): sql += " AND (name LIKE '{}')".format(like)

        return sm.executeGetList(self.corpdb, self.dbCursor, sql)

    def describeTable(self, table_name):
        """
         Describe the column names and column types of a table.

        Parameters
        ----------
        table_name: str
        Name of table to describe
 
        Returns
        -------
        list of lists 
        """

        sql = "PRAGMA table_info(%s);""" % (table_name)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql)

    def enable_table_keys(self, table):
        """
        No such feature for enabling keys in sqlite
        """
        pass

    def disable_table_keys(self, table):
        """
        No such feature for disabling keys in sqlite
        """
        pass

    def execute_get_list(self, usql):
        """
        Executes a given query, returns results as a list of lists

        Parameters
        ------------
        usql: str
            SELECT sql statement to execute

        Returns
        ------------
        List of list
        """
        return sm.executeGetList(self.corpdb, self.dbCursor, usql)


    def execute_get_SSCursor(self, usql):
        """
        No such feautre as using SSCursor for iterating over large returns. execute_get_list will be called in this case.
        """
        return sm.executeGetList(self.corpdb, self.dbCursor, usql) 

    def execute_write_many(self, sql, rows):
        """
        Executes the given insert query
        
        Parameters
        ---------
        sql: string
            Insert statement
        rows: list
            List of rows to insert into table 
        
        """
        sm.executeWriteMany(self.corpdb, self.dbConn, sql, rows, writeCursor=self.dbConn.cursor())

    def execute(self, sql):
        """
        Executes a given query

        Parameters
        ------------
        sql: str
            
        Returns
        ------------
        True or False depending on the success of query execution
        """
        return sm.execute(self.corpdb, self.dbConn, sql)

    def standardizeTable(self, table, collate, engine, charset, use_unicode):
        """
        All of these (collation sequence, charset and unicode) are assigned when creating the sqlite database. No such thing as 'engine' in sqlite.
        """
        pass

    def tableExists(self, table_name):
        """
        Checks whether a table exists
        
        Parameters
        ------------
        table_name: str

        Returns
        ------------
        True or False
        """
        return sm.tableExists(self.corpdb, self.dbCursor, table_name)
        
    def primaryKeyExists(self, table_name, column_name):
        """
        Checks whether a primary key exists in table_name on column_name

        Parameters
        ------------
        table_name: str
        
        column_name: str

        Returns
        ------------
        True or False
        """
        return sm.primaryKeyExists(self.corpdb, self.dbCursor, table_name, column_name)

    def indexExists(self, table_name, column_name):
        """
        Checks whether an index (which is not a primary key) exists
        
        Parameters
        ------------
        table_name: str
        
        column_name: str

        Returns
        ------------
        True or False
        """
        return sm.indexExists(self.corpdb, self.dbCursor, table_name, column_name)

    def getTableColumnNameTypes(self, table_name):
        """
        return a dict of column names mapped to types

        Parameters
        -------------
        table_name: str

        Returns
        -------------
        Dict
        """
        sql = "PRAGMA table_info("+table_name+")"
        data = sm.executeGetList(self.corpdb, self.dbCursor, sql)
        dictionary = {}
        for row in data:
            dictionary[row[1]] = row[2]
        return dictionary    

    def viewTable(self, table_name):
        """
        Returns the first 5 rows of table (to inspect).

        Parameters
        ----------
        table_name : :obj:`str`
        Name of table to describe

        Returns
        -------
        list of lists
        """
        col_sql = "PRAGMA table_info(%s);" % (table_name)
        col_names = [col[1] for col in sm.executeGetList(self.corpdb, self.dbCursor, col_sql)]

        sql = "SELECT * FROM %s LIMIT 10" % (table_name)
        return [col_names] + list(sm.executeGetList(self.corpdb, self.dbCursor, sql))

    def getRandomFunc(self, random_seed):
        """
        Gives the right "random" function based on data engine used.

        Parameters
        ----------
        random_seed: float
        Seed value used to produce the random number

        Returns
        -------
        str

        """
        #RANDOM() function in SQLiye doesn't consume a seed value.
        return "RANDOM()"

    
