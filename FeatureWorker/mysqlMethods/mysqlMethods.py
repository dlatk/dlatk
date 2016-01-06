import sys, time, datetime, os, getpass
import MySQLdb
import re
#import oursql
import csv
from random import sample
from math import floor

#DB INFO:                                                                                         
HOST = 'localhost'
USER = getpass.getuser()
PASSWD = ''
MAX_ATTEMPTS = 5 #max number of times to try a query before exiting                               
MYSQL_ERROR_SLEEP = 4 #number of seconds to wait before trying a query again (incase there was a \

def warn(string):
    print >>sys.stderr, string

def oConnect(db):
    """ Connects to specified database. Returns tuple of (dbConn, dbCursor, dictCursor) """
    dbConn = None
    attempts = 0;
    while (1):
        try:
            dbConn = oursql.connect(host=HOST, user=USER, passwd=PASSWD, db=db)
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

def executeGetSSCursor(self, db, sql):
    """Executes a given query (ss cursor is good to iterate over for large returns)"""
    #_warn("SQL (SSCursor) QUERY: %s"% sql)
    ssCursor = dbConnect(db)[0].cursor(MySQLdb.cursors.SSCursor)
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
            ssCursor = dbConnect(db)[0].cursor(MySQLdb.cursors.SSCursor)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return ssCursor

def dbConnect(db):
    """ Connects to specified database. Returns tuple of (dbConn, dbCursor, dictCursor) """
    dbConn = None
    attempts = 0;
    while (1):
        try:
            dbConn = MySQLdb.connect (
                host = HOST,
                user = USER,
                db = db,
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

def getTableColumnNames(db, table):
    """Returns a list of column names from a db table"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db)
    sql = "SELECT column_name FROM information_schema.columns WHERE table_schema='%s' AND table_name='%s'"%(db, table)
    columnNamesOfTable = executeGetList(db, dbCursor, sql)
    return map(lambda x: x[0], columnNamesOfTable)

def getTableColumnNamesTypes(db, table):
    """Returns a list of column names and types from a db table"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db)
    sql = "SELECT column_name, column_type FROM information_schema.columns WHERE table_schema='%s' AND table_name='%s'"%(db, table)
    return executeGetList(db, dbCursor, sql)

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

def execute( db, dbCursor, sql, warnQuery=False):
    """Executes a given query"""
    if warnQuery:
        warn("SQL QUERY: %s"% sql)
    attempts = 0;
    while (1):
        try:
            dbCursor.execute(sql)
            break
        except MySQLdb.Error, e:
            attempts += 1
            warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP)
            (dbConn, dbCursor, dictCursor) = dbConnect(db)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return True

def qExecute( db, sql, warnQuery=False):
    """performs the db connect and execute in the same call"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db)    
    return execute( db, dbCursor, sql, warnQuery)

def executeGetDict( db, dictCursor, sql, warnQuery=False):
    """Executes a given query, returns results as a list of dicts"""
    if warnQuery:
        warn("SQL (DictCursor) QUERY: %s"% sql)
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
            (dbConn, dbCursor, dictCursor) = dbConnect(db)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return data

def executeGetList( db, dbCursor, sql, warnQuery=False):
    """Executes a given query, returns results as a list of lists"""
    if warnQuery:
        warn("SQL QUERY: %s"% sql)
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
            (dbConn, dbCursor, dictCursor) = dbConnect(db)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return data

def executeGetList1( db, dbCursor, sql, warnQuery=False):
    """Executes a given query, expecting one resulting column. Returns results as a list"""
    return map(lambda x:x[0], executeGetList( db, dbCursor, sql, warnQuery))

def qExecuteGetList( db, sql, warnQuery=False):
    """performs the db connect and execute in the same call"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db)    
    return executeGetList( db, dbCursor, sql, warnQuery)

def qExecuteGetList1( db, sql, warnQuery=False):
    """performs the db connect and execute in the same call, equivalent to executeGetList1"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db)    
    return executeGetList1( db, dbCursor, sql, warnQuery)

def executeWrite( db, dbConn, sql, row, writeCursor=None , warnQuery=False):
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
            (dbConn, dbCursor, dictCursor) = dbConnect(db)
            writeCursor = dbConn.cursor()
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return writeCursor

def doesTableExist( db, table ):
    sql = 'show tables in %s'%db
    tables = qExecuteGetList1(db, sql)
    return table in tables

def executeWriteMany( db, dbConn, sql, rows, writeCursor=None, warnQuery=False):
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
            (dbConn, dbCursor, dictCursor) = dbConnect(db)
            writeCursor = dbConn.cursor()
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return writeCursor

def qExecuteWriteMany( db, sql, rows, writeCursor=None, warnQuery=False):
    """Executes a write query"""
    (dbConn, dbCursor, dictCursor) = dbConnect(db)
    return executeWriteMany(db, dbConn, sql, rows, writeCursor, warnQuery)

def gen_clone_query (conn, src_tbl, dst_tbl):
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

def getColumnNamesAndTypes( db, table ):
    """returns a dict of column names mapped to types"""
    sql = """SELECT column_name, column_type FROM information_schema.columns where TABLE_SCHEMA = '%s' AND TABLE_NAME \
= '%s'"""%(db, table)
    return dict(qExecuteGetList(db, sql))


def cloneExactTable( db, sourceTableName, destinationTableName ):
    warn("making TABLE %s, an exact copy of TABLE %s..."%(destinationTableName, sourceTableName))

    warn("connecting to DATABASE %s..."%(db))
    (dbConn, dbCursor, dictCursor) = dbConnect(db)

    warn("cloning structure of table...")
    cloneQuery = gen_clone_query( dbConn, sourceTableName, destinationTableName )
    execute(db, dbCursor, cloneQuery, True)
    
    warn("populating newly created table, TABLE %s"%(destinationTableName))
    populateQuery = "INSERT INTO " + destinationTableName + " SELECT * FROM " + sourceTableName
    execute(db, dbCursor, populateQuery, True)

    warn("finished cloning table!")

def randomSubsetTable( db, sourceTableName, destinationTableName, keyField, percentToSubset=.10, distinct=True ):
    warn("making TABLE %s, a %2.2f percent random subset of TABLE %s on unique key %s..."%(destinationTableName, percentToSubset, sourceTableName, keyField))

    warn("connecting to DATABASE %s..."%(db))
    (dbConn, dbCursor, dictCursor) = dbConnect(db)

    warn("removing destination table if it exists...")
    sql = 'DROP TABLE IF EXISTS %s'%(destinationTableName)
    execute(db, dbCursor, sql, True)

    warn("cloning structure of table...")
    sql = 'CREATE TABLE %s LIKE %s'%(destinationTableName, sourceTableName)
    execute(db, dbCursor, sql, True)
    
    isDistinctText = ' distinct' if distinct else ''
    warn('grabbing a%s subset (%2.6f percent) of the keys on which to base the new table'%(isDistinctText, 100*percentToSubset))
    sql = 'SELECT DISTINCT(%s) FROM %s'%(keyField, sourceTableName) if distinct else 'SELECT %s FROM %s'%(keyField, sourceTableName)
    uniqueKeyList = executeGetList1(db, dbCursor, sql, True)
    
    warn(str(uniqueKeyList[1:5]))

    newKeys = sample(uniqueKeyList, int(floor(len(uniqueKeyList)*percentToSubset)))
    newKeys = map(str, newKeys)

    warn("populating newly created table, TABLE %s"%(destinationTableName))
    populateQuery = "INSERT INTO %s SELECT * FROM %s WHERE %s IN (%s)"%(destinationTableName, sourceTableName, keyField, ','.join(newKeys))
    execute(db, dbCursor, populateQuery, False)

    warn("finished making TABLE %s, a %2.2f percent random subset of TABLE %s on unique key %s!"%(destinationTableName, percentToSubset, sourceTableName, keyField)) 
    

def writeTableToCSV(db, tableName, outputfile, sql_extra=None):
    (cur, conn, dcur) = dbConnect(db)
    sql = 'SELECT * FROM %s'%tableName
    if sql_extra:
        sql += ' ' + sql_extra
    dictRows = executeGetDict(db, dcur, sql)
    if dictRows:
        csvOut = csv.DictWriter(open(outputfile, 'w'), fieldnames=dictRows[0].keys())
        csvOut.writeheader()
        for dictionary in dictRows:
            csvOut.writerow(dictionary)
    else:
        raise Exception('No Results Returned From SQL Query, no output generated.')
