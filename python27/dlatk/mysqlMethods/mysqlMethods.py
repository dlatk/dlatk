import sys, time, datetime, os, getpass
import MySQLdb
import re
#import oursql
import csv
from random import sample
from math import floor

from dlatk.fwConstants import MAX_ATTEMPTS, MYSQL_ERROR_SLEEP, MYSQL_HOST, DEF_ENCODING, MAX_SQL_PRINT_CHARS, DEF_UNICODE_SWITCH, warn

#DB INFO:
USER = getpass.getuser()
PASSWD = ''                             
HOST = '127.0.0.1'

def executeGetSSCursor(db, sql, warnMsg = True, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH,host=HOST):
    """Executes a given query (ss cursor is good to iterate over for large returns)"""
    if warnMsg: 
        warn("SQL (SSCursor) QUERY: %s"% sql[:MAX_SQL_PRINT_CHARS])
    ssCursor = dbConnect(db, 
                         charset=charset, 
                         use_unicode=use_unicode,
                         host = host,
                         )[0].cursor(MySQLdb.cursors.SSCursor)
    data = []
    attempts = 0;
    while (1):
        try:
            ssCursor.execute(sql)
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP*attempts**2)
            ssCursor = dbConnect(db, charset=charset, use_unicode=use_unicode)[0].cursor(MySQLdb.cursors.SSCursor)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return ssCursor

def dbConnect(db, host=MYSQL_HOST, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """ Connects to specified database. Returns tuple of (dbConn, dbCursor, dictCursor) """
    dbConn = None
    attempts = 0;
    while (1):
        try:
            dbConn = MySQLdb.connect (
                host = host,
                user = USER,
                db = db,
                charset = charset,
                use_unicode = use_unicode, 
                read_default_file = "~/.my.cnf"
            )
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Connect ERROR on db:%s\n%s\n (%d attempt)"% (db, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    dbCursor = dbConn.cursor()
    dictCursor = dbConn.cursor(MySQLdb.cursors.DictCursor)
    return dbConn, dbCursor, dictCursor

def getTableColumnNames(db, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Returns a list of column names from a db table"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
    sql = "SELECT column_name FROM information_schema.columns WHERE table_schema='%s' AND table_name='%s'"%(db, table)
    columnNamesOfTable = executeGetList(db, dbCursor, sql)
    return map(lambda x: x[0], columnNamesOfTable)

def getTableColumnNamesTypes(db, table, charset=DEF_ENCODING):
    """Returns a list of column names and types from a db table"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
    sql = "SELECT column_name, column_type FROM information_schema.columns WHERE table_schema='%s' AND table_name='%s'"%(db, table)
    return executeGetList(db, dbCursor, sql, charset=charset, use_unicode=use_unicode)

def getTableColumnNameIndices(db, table, colNamesOfNote):
    """Returns a list of column indices pertaining to the colNames specified"""
    indexList = [None] * len(colNamesOfNote)
    columnNamesOfTable = getTableColumnNames(db, table)
    kk = 0
    for columnName in columnNamesOfTable:
        ii = 0
        for colOfNote in colNamesOfNote:
            if columnName == colOfNote:
                indexList[ii] = kk
            ii += 1
        kk += 1
    return indexList

def execute( db, dbCursor, sql, warnQuery=True, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a given query"""
    if warnQuery:
        warn("SQL QUERY: %s"% sql[:MAX_SQL_PRINT_CHARS])
    attempts = 0;
    while (1):
        try:
            dbCursor.execute(sql)
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return True

def qExecute( db, sql, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """performs the db connect and execute in the same call"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)    
    return execute( db, dbCursor, sql, warnQuery, charset=charset, use_unicode=use_unicode)

def executeGetDict( db, dictCursor, sql, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a given query, returns results as a list of dicts"""
    if warnQuery:
        warn("SQL (DictCursor) QUERY: %s"% sql[:MAX_SQL_PRINT_CHARS])
    data = []
    attempts = 0;
    while (1):
        try:
            dictCursor.execute(sql)
            data = dictCursor.fetchall()
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return data

def executeGetList( db, dbCursor, sql, warnQuery=True, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a given query, returns results as a list of lists"""
    if warnQuery:
        warn("SQL QUERY: %s"% sql[:MAX_SQL_PRINT_CHARS])
    data = []
    attempts = 0;
    while (1):
        try:
            dbCursor.execute(sql)
            data = dbCursor.fetchall()
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return data

def executeGetList1( db, dbCursor, sql, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a given query, expecting one resulting column. Returns results as a list"""
    return map(lambda x:x[0], executeGetList( db, dbCursor, sql, warnQuery, charset=charset, use_unicode=use_unicode))

def qExecuteGetList( db, sql, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """performs the db connect and execute in the same call"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)    
    return executeGetList( db, dbCursor, sql, warnQuery, charset=charset, use_unicode=use_unicode)

def qExecuteGetList1( db, sql, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """performs the db connect and execute in the same call, equivalent to executeGetList1"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)    
    return executeGetList1( db, dbCursor, sql, warnQuery, charset=charset, use_unicode=use_unicode)

def executeWrite( db, dbConn, sql, row, writeCursor=None , warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a write query"""
    if warnQuery:
        warn("SQL (write many) QUERY: %s"% sql)                                                 
    if not writeCursor:
        writeCursor = dbConn.cursor()
    attempts = 0;
    while (1):
        try:
            writeCursor.execute(sql, row)
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
            writeCursor = dbConn.cursor()
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return writeCursor

def doesTableExist( db, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    sql = 'show tables in %s'%db
    tables = qExecuteGetList1(db, sql, charset=charset, use_unicode=use_unicode)
    return table in tables

def executeWriteMany( db, dbConn, sql, rows, writeCursor=None, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a write query"""
    if warnQuery:
        warn("SQL (write many) QUERY: %s"% sql)                                                 
    if not writeCursor:
        writeCursor = dbConn.cursor()
    attempts = 0;
    while (1):
        try:
            writeCursor.executemany(sql, rows)
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
            writeCursor = dbConn.cursor()
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return writeCursor

def qExecuteWriteMany(db, sql, rows, writeCursor=None, warnQuery=False, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Executes a write query"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)
    return executeWriteMany(db, dbConn, sql, rows, writeCursor, warnQuery, charset=charset, use_unicode=use_unicode)

def gen_clone_query(conn, src_tbl, dst_tbl):
    try:
        cursor = conn.cursor ( )
        cursor.execute ("SHOW CREATE TABLE " + src_tbl)
        row = cursor.fetchone ( )
        cursor.close ( )
        if row == None:
            query = None
        else:
            # Replace src_tbl with dst_tbl in the CREATE TABLE statement
            query = re.sub ("CREATE TABLE .*`" + src_tbl + "`",
                            "CREATE TABLE `" + dst_tbl + "`",
                            row[1])
    except:
        query = None
    return query

def getColumnNamesAndTypes( db, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """returns a dict of column names mapped to types"""
    sql = """SELECT column_name, column_type FROM information_schema.columns where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'"""%(db, table)
    return dict(qExecuteGetList(db, sql, charset=charset, use_unicode=use_unicode))

def getTableEncoding(db, table=None, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """returns the encoding of a given table as a string"""
    if table: 
        sql = """SELECT CCSA.character_set_name FROM information_schema.`TABLES` T, information_schema.`COLLATION_CHARACTER_SET_APPLICABILITY` CCSA WHERE CCSA.collation_name = T.table_collation AND T.table_schema = '%s' AND T.table_name = '%s' """ % (db, table)
    else:
        sql = """SELECT default_character_set_name FROM information_schema.SCHEMATA WHERE schema_name = '%s'""" % (db)
    return str(qExecuteGetList1(db, sql, charset=charset, use_unicode=use_unicode)[0])

def cloneExactTable(db, sourceTableName, destinationTableName, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    warn("making TABLE %s, an exact copy of TABLE %s..."%(destinationTableName, sourceTableName))

    warn("connecting to DATABASE %s..."%(db))
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)

    warn("cloning structure of table...")
    cloneQuery = gen_clone_query( dbConn, sourceTableName, destinationTableName )
    execute(db, dbCursor, cloneQuery, True, charset=charset, use_unicode=use_unicode)
    
    warn("populating newly created table, TABLE %s"%(destinationTableName))
    populateQuery = "INSERT INTO " + destinationTableName + " SELECT * FROM " + sourceTableName
    execute(db, dbCursor, populateQuery, True, charset=charset, use_unicode=use_unicode)

    warn("finished cloning table!")

def randomSubsetTable( db, sourceTableName, destinationTableName, keyField, percentToSubset=.10, distinct=True, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    warn("making TABLE %s, a %2.2f percent random subset of TABLE %s on unique key %s..."%(destinationTableName, percentToSubset, sourceTableName, keyField))

    warn("connecting to DATABASE %s..."%(db))
    (dbConn, dbCursor, dictCursor) = dbConnect(db, charset=charset, use_unicode=use_unicode)

    warn("removing destination table if it exists...")
    sql = 'DROP TABLE IF EXISTS %s'%(destinationTableName)
    execute(db, dbCursor, sql, True, charset=charset, use_unicode=use_unicode)

    warn("cloning structure of table...")
    sql = 'CREATE TABLE %s LIKE %s'%(destinationTableName, sourceTableName)
    execute(db, dbCursor, sql, True, charset=charset, use_unicode=use_unicode)
    
    isDistinctText = ' distinct' if distinct else ''
    warn('grabbing a%s subset (%2.6f percent) of the keys on which to base the new table'%(isDistinctText, 100*percentToSubset))
    sql = 'SELECT DISTINCT(%s) FROM %s'%(keyField, sourceTableName) if distinct else 'SELECT %s FROM %s'%(keyField, sourceTableName)
    uniqueKeyList = executeGetList1(db, dbCursor, sql, True, charset=charset, use_unicode=use_unicode)
    
    warn(str(uniqueKeyList[1:5]))

    newKeys = sample(uniqueKeyList, int(floor(len(uniqueKeyList)*percentToSubset)))
    newKeys = map(str, newKeys)

    warn("populating newly created table, TABLE %s"%(destinationTableName))
    populateQuery = "INSERT INTO %s SELECT * FROM %s WHERE %s IN (%s)"%(destinationTableName, sourceTableName, keyField, ','.join(newKeys))
    execute(db, dbCursor, populateQuery, False, charset=charset, use_unicode=use_unicode)

    warn("finished making TABLE %s, a %2.2f percent random subset of TABLE %s on unique key %s!"%(destinationTableName, percentToSubset, sourceTableName, keyField)) 
    

def writeTableToCSV(db, tableName, outputfile, sql_extra=None, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    (cur, conn, dcur) = dbConnect(db, charset=charset, use_unicode=use_unicode)
    sql = 'SELECT * FROM %s'%tableName
    if sql_extra:
        sql += ' ' + sql_extra
    dictRows = executeGetDict(db, dcur, sql, charset=charset, use_unicode=use_unicode)
    if dictRows:
        csvOut = csv.DictWriter(open(outputfile, 'w'), fieldnames=dictRows[0].keys())
        csvOut.writeheader()
        for dictionary in dictRows:
            csvOut.writerow(dictionary)
    else:
        raise Exception('No Results Returned From SQL Query, no output generated.')

## TABLE MAINTENANCE ##
def optimizeTable(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Optimizes the table -- good after a lot of deletes"""
    sql = """OPTIMIZE TABLE %s """%(table)
    return execute(db, dbCursor, sql, charset=charset, use_unicode=use_unicode) 

def disableTableKeys(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Disable keys: good before doing a lot of inserts"""
    sql = """ALTER TABLE %s DISABLE KEYS"""%(table)
    return execute(db, dbCursor, sql, charset=charset, use_unicode=use_unicode) 

def enableTableKeys(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Enables the keys, for use after inserting (and with keys disabled)"""
    sql = """ALTER TABLE %s ENABLE KEYS"""%(table)
    return execute(db, dbCursor, sql, charset=charset, use_unicode=use_unicode) 

## Table Meta Info ##
def tableExists(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    sql = """show tables like '%s'""" % table
    if executeGetList(db, dbCursor, sql, charset=charset, use_unicode=use_unicode):
        return True
    else:
        return False

def getTableDataLength(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Returns the data length for the given table"""
    sql = """SELECT DATA_LENGTH FROM information_schema.tables where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'""" % (db, table)
    return executeGetList(db, dbCursor, sql, charset=charset, use_unicode=use_unicode)[0]

def getTableIndexLength(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """Returns the data length for the given table"""
    sql = """SELECT INDEX_LENGTH FROM information_schema.tables where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'""" % (db, table)
    return executeGetList(db, dbCursor, sql, charset=charset, use_unicode=use_unicode)[0]

def getTableColumnNameTypes(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """returns a dict of column names mapped to types"""
    sql = """SELECT column_name, column_type FROM information_schema.columns where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'"""%(db, table)
    return dict(executeGetList(db, dbCursor, sql, charset=charset, use_unicode=use_unicode))


def getTableColumnNameList(db, dbCursor, table, charset=DEF_ENCODING, use_unicode=DEF_UNICODE_SWITCH):
    """returns a dict of column names mapped to types"""
    sql = """SELECT column_name FROM information_schema.columns where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s' ORDER BY ORDINAL_POSITION"""%(db, table)
    return [x[0] for x in executeGetList(db, dbCursor, sql, charset=charset, use_unicode=use_unicode)]

#our_sql
def oConnect(db):
    """ Connects to specified database. Returns tuple of (dbConn, dbCursor, dictCursor) """
    dbConn = None
    attempts = 0;
    while (1):
        try:
            dbConn = oursql.connect(host=MYSQL_HOST, user=USER, passwd=PASSWD, db=db)
            break
        except e:
            attempts += 1
            warn(" *OUSQL Connect ERROR on db:%s\n%s\n (%d attempt)"% (db, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    dbCursor = dbConn.cursor(try_plain_query=False)
    dictCursor = dbConn.cursor(oursql.DictCursor)
    return dbConn, dbCursor, dictCursor

def oTestQueryIteration(sql, cur):
    b = datetime.datetime.now()
    print 'running execute statement...'
    cur.execute(sql)
    e = datetime.datetime.now()
    print "...finished execution! Took %d seconds."%(e-b).total_seconds()
    print "iterating  through rows..."
    b = datetime.datetime.now()
    counter = 0
    while True:
        row = cur.fetchone()
        if not row: break
        if counter % 10000000 == 0:
            print row
        counter += 1
    e = datetime.datetime.now()
    print "...finished iterating! Took %d seconds."%(e-b).total_seconds()
    print counter

