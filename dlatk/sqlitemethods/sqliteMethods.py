"""
Sqlite interface based on the sqlite3 package
"""

import sys, time
import sqlite3
from dlatk.dlaConstants import MAX_ATTEMPTS, SQLITE_ERROR_SLEEP, MAX_SQL_PRINT_CHARS, warn

def dbConnect(db):
	"""
	Creates connection with the database
	"""
	db = db + ".db"
	dbConn = None
	attempts = 0
	while(1):
		try:
			dbConn = sqlite3.connect(db)
			break
		except sqlite3.Error as err:
			attempts += 1
			warn(" *Sqlite Connect Error on db: %s\n%s\n (%d attempt)"%(db, err, attempts))
			time.sleep(SQLITE_ERROR_SLEEP)
			if(attempts > MAX_ATTEMPTS):
				sys.exit(1)		
	dbCursor = dbConn.cursor()
	return dbConn, dbCursor

def executeWriteMany(db, dbConn, sql, rows, writeCursor=None, warnQuery=False):
	"""Executes a write query"""
	if warnQuery:
		warn("SQL (write many) QUERY: %s"% sql)
	if not writeCursor:
		writeCursor = dbConn.cursor()
	attempts = 0
	while(1):
		try:	
			writeCursor.executemany(sql, rows)
			dbConn.commit()
			break
		except sqlite3.Error as err:
			attempts += 1
			warn(" *Sqlite Corpus DB Error on %s:\n%s (%d attempt)" % (sql, err, attempts))
			time.sleep(SQLITE_ERROR_SLEEP)
			dbConn, dbCursor = dbConnect(db)
			writeCursor = dbConn.cursor()
			if(attempts > MAX_ATTEMPTS):
				sys.exit(1)
	return writeCursor

def executeGetList(db, dbCursor, sql, warnQuery=False):
	"""Executes a SELECT query"""
	if warnQuery:
		warn("SQL Query: %s"% sql[:MAX_SQL_PRINT_CHARS])
	attempts = 0
	data = ""
	while(1):
		try:
			dbCursor.execute(sql)
			data = dbCursor.fetchall()
			break
		except sqlite3.Error as err:
			attempts += 1
			warn(" *Sqlite Corpus DB Error on %s:\n%s (%d attempt)" % (sql, err, attempts))
			time.sleep(SQLITE_ERROR_SLEEP)
			dbConn, dbCursor = dbConnect(db)
			writeCursor = dbConn.cursor()
			if(attempts > MAX_ATTEMPTS):
				sys.exit(1)
	return data

def execute(db, dbCursor, sql, warnQuery=True):
	"""Executes a given query"""
	if warnQuery:
		warn("SQL Query: %s" % sql[:MAX_SQL_PRINT_CHARS])
	attempts = 0
	while(1):
		try:
			dbCursor.execute(sql)
			break
		except sqlite3.Error as err:
			attempts += 1
			warn(" *Sqlite Corpus DB Error on %s:\n%s (%d attempt)" % (sql, err, attempts))
			time.sleep(SQLITE_ERROR_SLEEP)
			dbConn, dbCursor = dbConnect(db)
			if(attempts > MAX_ATTEMPTS):
				sys.exit(1)
	return True

def tableExists(db, dbCursor, table_name):
	"""Checks if table exists"""
	sql = """SELECT count(name) FROM sqlite_master WHERE type='table' AND name='%s'"""% table_name
	count = executeGetList(db, dbCursor, sql)
	if count[0][0] > 0:
		return True
	else:
		return False

def primaryKeyExists(db, dbCursor, table_name, column_name):
	"""Checks if primary key exists on a table's column"""
	sql = "PRAGMA table_info("+table_name+")"
	data = executeGetList(db, dbCursor, sql)
	for row in data:
		if row[1] == column_name and row[len(row)-1] == 1:
			print(row)
			print(row[1])
			print(column_name)
			print(row[len(row)-1])
			return True
	return False
	
def indexExists(db, dbCursor, table_name, column_name):
	"""Checks if index exists on a table's column"""
	sql = "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'"
	data = executeGetList(db, dbCursor, sql)
	for row in data:
		db_sql = row[len(row)-1].split(" ")
		db_col_name = db_sql[len(db_sql)-1][1:-1]
		if row[1] == table_name and db_col_name == column_name:
			return True
	return False
