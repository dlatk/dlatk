from abc import ABC, abstractmethod
from .dataEngine import DataEngine
from .. import dlaConstants as dlac
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
	data_engine: str
		Name of the data engine eg. mysql, sqlite
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
		return CreateTableQuery(table, self.data_engine)


class SelectQuery(Query):
	"""
	Class for building a SELECT query.

	Parameters
	------------
	from_table: str
		Name of the table from which records are to be fetched.
	data_engine: str
		Name of the database engine being used.
	"""

	def __init__(self, from_table, data_engine):
		super().__init__()
		self.sql = None
		self.fields = None
		self.where_conditions = None
		self.group_by_fields = None
		self.from_table = from_table
		self.data_engine = data_engine

	def set_fields(self, fields):
		"""
		Parameters
		------------
		fields: list
			List containing name of the columns
	
		Returns	
		------------
		SelectQuery object		
		"""
		self.fields = fields
		return self

	def where(self, where_conditions):
		self.where_conditions = where_conditions
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

	def execute_query(self):
		"""
		Executes sql query
	
		Returns
		------------
		List of lists
		"""
		self.sql = self.build_query()
		data = self.data_engine.execute_get_list(self.sql)
		if self.sql.lower().startswith("pragma"):
			for row in data:
				print(row)
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
			if self.group_by_fields:
				selectQuery += """ GROUP BY %s"""%(', '.join(self.group_by_fields))
			if self.where_conditions:
				if len(self.where_conditions)>0:
					selectQuery += """ WHERE """
					for f in self.where_conditions:
						selectQuery += str(f[0]) +"""='"""+ str(f[1])  +"""' AND """
					selectQuery = selectQuery[:-4]
			return selectQuery
			#return """SELECT %s FROM %s GROUP BY %s""" %(','.join(self.fields), self.from_table, ','.join(self.group_by_fields))

		if self.data_engine.db_type == "sqlite":
			if self.from_table == "information_schema.columns":
				for f in self.where_conditions:
					if f[0] == "table_name":
						table_name =  f[1]
					if f[0] == "column_name":
						self.column_name = f[1]					
				selectQuery = "PRAGMA table_info("+table_name+")"
				return selectQuery
				
			else:
				selectQuery = """SELECT %s FROM %s""" %(', '.join(self.fields), self.from_table)
				if self.group_by_fields:
					selectQuery += """ GROUP BY %s"""%(', '.join(self.group_by_fields))
				if self.where_conditions:
					if len(self.where_conditions)>0:
						selectQuery += """ WHERE """
						for f in self.where_conditions:
							selectQuery += str(f[0]) +"""='"""+ str(f[1])  +"""' AND """
						selectQuery = selectQuery[:-4]
				return selectQuery
			#return """SELECT ? FROM ? GROUP BY ?""" %(','.join(self.fields), self.from_table, ','.join(self.group_by_fields))


class InsertQuery(Query):

	
	def __init__(self, table, data_engine):
		super().__init__()
		self.sql = None
		self.values = None
		self.group_by_fields = None
		self.table = table
		self.data_engine = data_engine

	def set_values(self, values):
		self.values = values
		return self

	def build_query(self):
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
		if self.sql == None:
			self.sql = self.build_query()
		self.data_engine.execute_write_many(self.sql, insert_rows)



class DropQuery(Query):

	
	def __init__(self, table, data_engine):
		super().__init__()
		self.sql = None
		self.table = table
		self.data_engine = data_engine

	def build_query(self):
		dropQuery = """DROP TABLE IF EXISTS %s""" % self.table
		return dropQuery

	def execute_query(self):
		if self.sql == None:
			self.sql = self.build_query()
		self.data_engine.execute(self.sql)



class CreateTableQuery(Query):

	def __init__(self, table, data_engine):
		super().__init__()
		self.sql = None
		self.table = table
		self.data_engine = data_engine
		self.cols = None
		self.mul_keys = None
		self.char_set = None
		self.collation = None
		self.engine = None

	def add_columns(self, cols):
		"""
		List of Column objects
		"""
		self.cols = cols
		return self

	def add_mul_keys(self, keys):
		"""
		keys is a dict with name of the key and column name
		"""
		self.mul_keys = keys
		return self

	def set_character_set(self, char_set):
		self.char_set = char_set
		return self

	def set_collation(self, collation):
		self.collation = collation
		return self		

	def set_engine(self, engine):
		self.engine = engine
		return self

	def build_query(self):
		if self.data_engine.db_type == "mysql":
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
			createTable = """CREATE TABLE %s (""" % self.table 
			for col in self.cols:
				datatype = ""
				if col.is_unsigned():
					datatype += "UNSIGNED "
				datatype += col.get_datatype().split()[0]
				createTable  += """ %s""" % col.get_name()
				createTable += """ %s""" % datatype
				if col.is_primary_key()!=True and col.is_nullable()!=True:
					createTable += """ NOT NULL"""
				if col.is_auto_increment():
					pass
					#createTable += """ AUTOINCREMENT"""
				if col.is_primary_key():
					createTable += """ PRIMARY KEY"""			
				createTable += ""","""
				
	
			createTable = createTable[0:-1]
			createTable += """)"""
	
			#if self.char_set:
			#	createTable += """ CHARACTER SET %s""" % self.char_set
	
			#if self.collation:
			#	createTable += """ COLLATE %s""" % self.collation
	
			return createTable	

	def execute_query(self):
		self.sql = self.build_query()	
		success = self.data_engine.execute(self.sql)
		print("\nSUCCESSFULL QUERY: %s\n"%self.sql)
		if success and self.data_engine.db_type == "sqlite":
			print("\nINSIDE success and yes this is sqlite\n")
			if len(self.mul_keys)>0:
				for key in self.mul_keys:
					print("\n\nCreating index %s on table:%s, column:%s \n\n"%key[0], self.table, key[1])
					createIndex = """CREATE INDEX %s ON %s (%s)""" % (key[0], self.table, key[1])
					self.data_engine.execute(createIndex)


class Column():

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
		return self.column_name

	def get_datatype(self):
		return self.datatype		

	def is_primary_key(self):
		return self.primary_key

	def is_nullable(self):
		return self.nullable

	def is_auto_increment(self):
		return self.auto_increment

	def is_unsigned(self):
		return self.unsigned
