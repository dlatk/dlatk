from ..mysqlmethods import mysqlMethods as mm
from ..sqlitemethods import sqliteMethods as sm
from .. import dlaConstants as dlac
import sys
import csv

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

class SqliteDataEngine(DataEngine):
	def __init__(self, corpdb):
		super().__init__(corpdb)
		(self.dbConn, self.dbCursor) = sm.dbConnect(corpdb)

	def get_db_connection(self):
		"""
		Returns
		------------
		Database connection objects
		"""
		return self.dbConn, self.dbCursor, None

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

	def csvToTable(self, csv_file, table_name, column_description, ignoreLines=1):
		"""
		Loads a CSV file as a SQLite table to the database
		
		Parameters
		------------
		csv_file: str

		table_name: str
		
		column_description: str
		"""

		if self.tableExists(table_name):
			#FIXME - raise an exception instead
			print("A table by that name already exists in the database")
			sys.exit(1)

		createSQL = "CREATE TABLE {} {}".format(table_name, column_description)
		self.execute(createSQL)

		print("Importing data, reading {} file".format(csv_file))

		def chunks(data, rows=10000):
			"Divides the data into 10000 rows each"
			for i in range(0, len(data), rows):
				yield data[i:i+rows]

		with open(csv_file, 'r') as f:
			reader = csv.reader(f, delimiter=',')
			if ignoreLines > 0:
				for i in range(0, ignoreLines): 
					next(reader)
			data = list(reader)
			chunk_data = chunks(data) 
			num_columns = None
			for chunk in chunk_data:
				if not num_columns:
					num_columns = len(chunk[0])
					values_str = "(" + ",".join(["?"] * num_columns) + ")"
				insertQuery = "INSERT INTO {} VALUES {}".format(table_name, values_str)
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
		
		selectQuery = "SELECT * FROM {}".format(table)
		self.dbCursor.execute(selectQuery)
		header = [i[0] for i in self.dbCursor.description]
		with open(csv_file, 'w') as f:
			csv_writer = csv.writer(f, quoting=quoting)
			csv_writer.writerow(header)
			csv_writer.writerows(self.dbCursor)

		return
