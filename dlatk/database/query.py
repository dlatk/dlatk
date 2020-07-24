from abc import ABC, abstractmethod
from .dataEngine import DataEngine
import sys

class Query(ABC):
	"""
	Abstract class. Base class for the different query building classes like SelectQuery
	"""
	def __init__(self):
		pass

	@abstractmethod
	def build_query(self):
		pass

	@abstractmethod
	def execute_query(self):
		pass


class QueryBuilder():
	"""
	This class is directly used by DLATK classes for building queries.

	Parameters
	------------
	data_engine: object
		Object of the database engine being used i.e. either MySqlDataEngine or SqliteDataEngine.
	"""
	def __init__(self, data_engine):
		self.data_engine = data_engine

	def create_select_query(self, from_table):
		"""
		Parameters
		------------
		from_table: str
			Name of the table from which records are to be fetched.
		
		Returns
		------------
		Object of SelectQuery class	
		"""
		return SelectQuery(from_table, self.data_engine)


	def create_insert_query(self, into_table):
		"""
		Parameters
		------------
		from_table: str
			Name of the table into which records are to be added.
		
		Returns
		------------
		Object of InsertQuery class	
		"""
		return InsertQuery(into_table, self.data_engine)

	def create_drop_query(self, table):
		"""
		Parameters
		------------
		from_table: str
			Name of the table which is to be droped.
		
		Returns
		------------
		Object of DropQuery class
		"""
		return DropQuery(table, self.data_engine)

	def create_createTable_query(self, table):
		"""
		Parameters
		------------
		table: str
			Name of the table which is to be created.
		
		Returns
		------------
		Object of createTable class
		"""
		return CreateTableQuery(table, self.data_engine)


class SelectQuery(Query):
	"""
	Class for building a SELECT query.

	Parameters
	------------
	from_table: str
		Name of the table from which records are to be fetched.
	
	data_engine: object
		Object of the database engine being used i.e. either MySqlDataEngine or SqliteDataEngine.
	"""

	def __init__(self, from_table, data_engine):
		super().__init__()
		self.sql = None
		self.fields = None
		self.where_conditions = "" 
		self.group_by_fields = None
		self.order_by_fields = None
		self.limit = None
		self.from_table = from_table
		self.data_engine = data_engine

	def set_fields(self, fields):
		"""
		Parameters
		------------
		fields: list
			List containing name of the columns to fetch
	
		Returns	
		------------
		SelectQuery object		
		"""
		self.fields = fields
		return self

	def where(self, where_conditions):
		"""
		Parameters
		------------
		where_conditions: str
			where clauses as a string 

		Returns
		------------
		SelectQuery object
		"""
		self.where_conditions += where_conditions
		return self

	def group_by(self, group_by_fields):
		"""
		Parameters
		------------
		group_by_fields: list
			List containing name of the fields which should be used for grouping.
	
		Returns	
		------------
		SelectQuery object		
		"""
		self.group_by_fields = group_by_fields
		return self

	def order_by(self, order_by_fields):
		"""
		Parameters
		----------
		order_by_fields: list
			List containing tuples of form (field_name, 'ASC'|'DESC')

		Returns
		-------
		SelectQuery object
		"""
		self.order_by_fields = ['{} {}'.format(field_name, asc_desc) for field_name, asc_desc in order_by_fields]
		return self

	def set_limit(self, limit):
		self.limit = limit
		return self

	def toString(self):
		return self.build_query()

	def execute_query(self):
		"""
		Executes sql query
	
		Returns
		------------
		List of lists
		"""
		self.sql = self.build_query()
		data = self.data_engine.execute_get_list(self.sql)
		if self.sql.lower().startswith("pragma") and self.fields[0] == "column_type" and len(self.fields) == 1:
			for row in data:
				if row[1] == self.column_name:
					return [[row[2]]]					

		return data 

	def build_query(self):
		"""
		Builds a sql query based on the type of db
	
		Returns
		------------
		str: a built select query
		"""
		if self.data_engine.db_type == "mysql":
			selectQuery = """SELECT %s FROM %s""" %(', '.join(self.fields), self.from_table)
			if self.where_conditions:
				if len(self.where_conditions)>0:
					if not self.where_conditions.lstrip().lower().startswith("where"):
						selectQuery += """ WHERE """ + self.where_conditions
					else:
						selectQuery += " " + self.where_conditions
					#for f in self.where_conditions:
					#	selectQuery += str(f[0]) +"""='"""+ str(f[1])  +"""' AND """
					#selectQuery = selectQuery[:-4]
			if self.group_by_fields:
				selectQuery += """ GROUP BY %s"""%(', '.join(self.group_by_fields))
			if self.order_by_fields:
				selectQuery += """ ORDER BY {}""".format(', '.join(self.order_by_fields))
			if self.limit:
				selectQuery += """ LIMIT {}""".format(str(self.limit))
			return selectQuery

		if self.data_engine.db_type == "sqlite":
			if self.from_table == "information_schema.columns":
				for f in self.where_conditions.split(" "):
					# extract table name from the where condition
					if f.startswith("table_name"):
						table_name = f.split("=")[1]
						# strip quotes from the table name
						table_name = table_name[1:-1]
					# extract column name from the where condition
					if f.startswith("column_name"):
						self.column_name = f.split("=")[1]
						# strip quotes from the column name
						self.column_name = self.column_name[1:-1]
				#for f in self.where_conditions:
				#	if f[0] == "table_name":
				#		table_name =  f[1]
				#	if f[0] == "column_name":
				#		self.column_name = f[1]					
				selectQuery = "PRAGMA table_info("+table_name+")"
				return selectQuery
				
			else:
				selectQuery = """SELECT %s FROM %s""" %(', '.join(self.fields), self.from_table)
				if self.where_conditions:
					if len(self.where_conditions)>0:
						if not self.where_conditions.lstrip().lower().startswith("where"):
							selectQuery += """ WHERE """ + self.where_conditions
						else:
							selectQuery += " " + self.where_conditions
						#selectQuery += """ WHERE """
						#for f in self.where_conditions:
						#	selectQuery += str(f[0]) +"""='"""+ str(f[1])  +"""' AND """
						#selectQuery = selectQuery[:-4]
				if self.group_by_fields:
					selectQuery += """ GROUP BY %s"""%(', '.join(self.group_by_fields))
				if self.order_by_fields:
					selectQuery += """ ORDER BY {}""".format(', '.join(self.order_by_fields))
				if self.limit:
					selectQuery += """ LIMIT {}""".format(str(self.limit))
				return selectQuery


class InsertQuery(Query):
	"""
	Class for building an INSERT query.

	Parameters
	------------
	table: str
		Name of the table from which is to be created.
	data_engine: object
		Object of the database engine being used i.e. either MySqlDataEngine or SqliteDataEngine.
	"""
	def __init__(self, table, data_engine):
		super().__init__()
		self.sql = None
		self.values = None
		self.group_by_fields = None
		self.table = table
		self.data_engine = data_engine

	def set_values(self, values):
		"""
		Parameters
		------------
		values: list of tuples
			tuples containing column name and value for that column to insert into table

		Returns
		------------
		InsertQuery object
		"""
		self.values = values
		return self

	def build_query(self):
		"""
		Builds a INSERT query based on the type of db
	
		Returns
		------------
		str: a built INSERT query
		"""
		if self.data_engine.db_type == "mysql":
			fields = ""
			vals = ""
			numOfValues = len(self.values)
			for f in self.values:
				fields += f[0] + ", "
			fields = fields[0:-2]
			for v in self.values:
				if len(str(v[1]).strip()) == 0:
					vals += "%s ,"
				else:
					vals += "'" + str(v[1])+ "',"
			vals = vals[0:-2]
			insertQuery = """INSERT INTO """+ self.table +""" ("""+ fields +""") values ("""+ vals +""")"""
			return insertQuery
		if self.data_engine.db_type == "sqlite":
			fields = ""
			vals = ""
			numOfValues = len(self.values)
			for f in self.values:
				fields += f[0] + ", "
			fields = fields[0:-2]
			for v in self.values:
				if len(str(v[1]).strip()) == 0:
					vals += "? ,"
				else:
					vals += "'" + str(v[1]) + "',"
			vals = vals[0:-2]
			insertQuery = """INSERT INTO """+ self.table +""" ("""+ fields +""") values ("""+ vals + """)"""
			return insertQuery

	def execute_query(self, insert_rows):
		"""
		Executes INSERT query

		Parameters
		------------
		insert_rows: values of columns for each row to be inserted in the table
		"""
		if self.sql == None:
			self.sql = self.build_query()
		self.data_engine.execute_write_many(self.sql, insert_rows)



class DropQuery(Query):
	"""
	Class for building a DROP query.

	Parameters
	------------
	table: str
		Name of the table to be droped.
	data_engine: object
		Object of the database engine being used i.e. either MySqlDataEngine or SqliteDataEngine.
	"""
	def __init__(self, table, data_engine):
		super().__init__()
		self.sql = None
		self.table = table
		self.data_engine = data_engine

	def build_query(self):
		"""
		Builds a DROP query based on the type of db
	
		Returns
		------------
		str: a built DROP query
		"""
		dropQuery = """DROP TABLE IF EXISTS %s""" % self.table
		return dropQuery

	def execute_query(self):
		"""
		Executes the DROP query
		"""
		if self.sql == None:
			self.sql = self.build_query()
		self.data_engine.execute(self.sql)



class CreateTableQuery(Query):
	"""
	Class for building a CREATE query.

	Parameters
	------------
	table: str
		Name of the table to be created.
	data_engine: object
		Object of the database engine being used i.e. either MySqlDataEngine or SqliteDataEngine.
	"""
	def __init__(self, table, data_engine):
		super().__init__()
		self.sql = None
		self.table = table
		self.likeTable = None
		self.data_engine = data_engine
		self.cols = None
		self.mul_keys = None
		self.char_set = None
		self.collation = None
		self.engine = None

	def add_columns(self, cols):
		"""
		Parameters
		------------
		cols: list
			List of Column objects

		Returns
		------------
		CreateTableQuery object
		"""
		self.cols = cols
		return self

	def add_mul_keys(self, keys):
		"""
		Parameters
		------------
		keys: dict
			dict containing name of the key and column name

		Returns
		------------
		CreateTableQuery object
		"""
		self.mul_keys = keys
		return self

	def set_character_set(self, char_set):
		"""
		Parameters
		------------
		char_set: str
			character encoding

		Returns
		------------
		CreateTableQuery object
		"""
		self.char_set = char_set
		return self

	def set_collation(self, collation):
		"""
		Parameters
		------------
		collation: str
			collation sequence for the character set

		Returns
		------------
		CreateTableQuery object
		"""
		self.collation = collation
		return self		

	def set_engine(self, engine):
		"""
		Parameters
		------------
		engine: str
			type of storage engine

		Returns
		------------
		CreateTableQuery object
		"""
		self.engine = engine
		return self

	def like(self, oldTable):
		"""
		Create a table similar to the oldTable
		"""
		self.likeTable = oldTable
		return self

	def execute_query(self):
		"""
		Executes a CREATE table query and also creates indexs on the table after the table has been created (for sqlite)
		"""
		self.sql = self.build_query()	
		success = self.data_engine.execute(self.sql)
		if success==True and self.data_engine.db_type == "sqlite":
			if self.mul_keys:
				for key in self.mul_keys:
					print("""\n\nCreating index {0} on table:{1}, column:{2} \n\n""".format(key[0], self.table, key[1]))
					createIndex = """CREATE INDEX %s ON %s (%s)""" % (key[0]+self.table[4:], self.table, key[1])
					if "meta" not in self.table:
						self.data_engine.execute(createIndex)

	def build_query(self):
		"""
		Builds a sql query based on the type of db
	
		Returns
		------------
		str: a built select query
		"""
		if self.data_engine.db_type == "mysql":
			if self.likeTable != None:
				return """CREATE TABLE %s LIKE %s"""%(self.table, self.likeTable)
			createTable = """CREATE TABLE %s (""" % self.table
			for col in self.cols:
				datatype = col.get_datatype()
				createTable  += """ %s""" % col.get_name()
				createTable += """ %s""" % datatype
				if col.is_unsigned():
					createTable += """ UNSIGNED"""
				if col.is_nullable()!=True:
					createTable += """ NOT NULL"""
				if col.is_auto_increment():
					createTable += """ AUTO_INCREMENT"""
				if col.is_primary_key():
					createTable += """ PRIMARY KEY"""			
				createTable += ""","""
				
			if self.mul_keys!=None and len(self.mul_keys)>0:
				for key in self.mul_keys:
					createTable += """ KEY `%s` (`%s`),""" % (key[0], key[1])
	
			createTable = createTable[0:-1]
			createTable += """)"""
	
			if self.char_set:
				createTable += """ CHARACTER SET %s""" % self.char_set
	
			if self.collation:
				createTable += """ COLLATE %s""" % self.collation
	
			if self.engine:
				createTable += """ ENGINE=%s""" % self.engine
	
			return createTable

		if self.data_engine.db_type == "sqlite":
			if self.likeTable != None:
				"""write code to extract the sql statement used to create the likeTable and change the name of the table to self.table"""
				sql = """SELECT sql FROM sqlite_master WHERE type='table' AND name='%s'""" % self.likeTable
				originalCreateQuery = self.data_engine.execute_get_list(sql)[0][0]
				return originalCreateQuery.replace(self.likeTable, self.table)
	
			createTable = """CREATE TABLE %s (""" % self.table 
			for col in self.cols:
				datatype = ""
				if col.is_unsigned():
					datatype += "UNSIGNED "
				datatype += col.get_datatype().split()[0]
				createTable  += """ %s""" % col.get_name()
				if col.get_name() == "id" and "BIGINT" or "INT" in datatype:
					datatype = "INTEGER"
				createTable += """ %s""" % datatype
				if col.is_primary_key()!=True and col.is_nullable()!=True:
					createTable += """ NOT NULL"""
				if col.is_primary_key():
					createTable += """ PRIMARY KEY"""			
				#if col.is_auto_increment():
				#	createTable += """ AUTOINCREMENT"""
				createTable += ""","""
				
	
			createTable = createTable[0:-1]
			createTable += """)"""
	
			#if self.char_set:
			#	createTable += """ CHARACTER SET %s""" % self.char_set
	
			#if self.collation:
			#	createTable += """ COLLATE %s""" % self.collation
	
			return createTable	



class Column():
	"""
	This class is directly used by DLATK classes for building queries.

	Parameters
	------------
	column_name: str
		
	dataype: str

	unsigned: bool
	
	primary_key: bool

	nullable: bool

	auto_increment: bool
	"""

	def __init__(self, column_name, datatype, unsigned=False, primary_key=False, nullable=True, auto_increment=False):
		self.column_name = column_name
		self.datatype = datatype
		self.unsigned = unsigned
		self.primary_key = primary_key
		self.nullable = nullable
		if self.primary_key:
			self.nullable = False
		self.auto_increment = auto_increment

	def get_name(self):
		"""
		Returns
		------------
		str: name of the column
		"""
		return self.column_name

	def get_datatype(self):
		"""
		Returns
		------------
		str: data type of the column
		"""
		return self.datatype		

	def is_primary_key(self):
		"""
		Returns
		------------
		bool: True if the column is primary key otherwise False
		"""
		return self.primary_key

	def is_nullable(self):
		"""
		Returns
		------------
		bool: True if the column is nullable(can have null values) otherwise False
		"""
		return self.nullable

	def is_auto_increment(self):
		"""
		Returns
		------------
		bool: True if column is auto increment
		"""
		return self.auto_increment

	def is_unsigned(self):
		"""
		Returns
		------------
		bool: True if column is unsigned
		"""
		return self.unsigned
