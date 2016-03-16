#!/usr/bin/python
###########################################################
## featureWorker.py
##
## Interface Module to extract features and create tables holding the features
##
## Contributors:
##
## TODO:
## -handle that mysql is not using mixed case (should we lowercase all features?)

import sys
import time
import MySQLdb

#infrastructure
import fwConstants as fwc
from mysqlMethods import mysqlMethods as mm 

##############################################################
### Class Definitions
##
#
class FeatureWorker(object):
    """Generic class for functions working with features"""
    def __init__(self, corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, lexicondb = fwc.DEF_LEXICON_DB, date_field=fwc.DEF_DATE_FIELD, wordTable = None):
        self.corpdb = corpdb
        self.corptable = corptable
        self.correl_field = correl_field
        self.mysql_host = mysql_host
        self.message_field = message_field
        self.messageid_field = messageid_field
        self.encoding = encoding
        (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(corpdb, host=mysql_host, charset=encoding)
        self.lexicondb = lexicondb
        self.wordTable = wordTable if wordTable else "feat$1gram$%s$%s$16to16"%(self.corptable, self.correl_field)

    ##PUBLIC METHODS#
    def getMessages(self, messageTable = None, where = None):
        """..."""
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s FROM %s"""% (self.messageid_field, self.message_field, messageTable)
        if where: msql += " WHERE " + where
        return mm.executeGetSSCursor(self.corpdb, msql)

    def getMessagesForCorrelField(self, cf_id, messageTable = None, warnMsg = True):
        """..."""
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s FROM %s WHERE %s = '%s'""" % (
            self.messageid_field, self.message_field, messageTable, self.correl_field, cf_id)
        #return self._executeGetSSCursor(msql, warnMsg)
        return mm.executeGetList(self.corpdb, self.dbCursor, msql, warnMsg)

    def getMessagesWithFieldForCorrelField(self, cf_id, extraField, messageTable = None, warnMsg = True):
        """..."""
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s, %s FROM %s WHERE %s = '%s'""" % (
            self.messageid_field, self.message_field, extraField, messageTable, self.correl_field, cf_id)
        #return self._executeGetSSCursor(msql, showQuery)
        return mm.executeGetList(self.corpdb, self.dbCursor, msql, warnMsg)

    def getNumWordsByCorrelField(self, where = ''):
        """..."""
        #assumes corptable has num_words field for each message
        #SELECT user_id, sum(num_words) FROM (SELECT user_id, num_words FROM messages GROUP BY message_id) as a GROUP BY user_id
        sql = """SELECT %s, sum(num_words) FROM (SELECT %s, num_words FROM %s """ % (self.correl_field, self.correl_field, self.corptable)
        if (where): sql += ' WHERE ' + where  
        sql += """ GROUP BY %s) as a """ % self.messageid_field 
        sql += """ GROUP BY %s """ % self.correl_field
        return mm.executeGetList(self.corpdb, self.dbCursor, sql)

    def getWordTable(self, corptable = None):
        if self.wordTable: return self.wordTable
        if not corptable:
            corptable  = self.corptable
        return "feat$1gram$%s$%s$16to16"%(corptable, self.correl_field)

    def get1gramTable(self):
        return "feat$1gram$%s$%s$16to16"%(self.corptable, self.correl_field)
        
    def getWordTablePOcc(self, pocc):
        return "feat$1gram$%s$%s$16to16$%s"%(self.corptable, self.correl_field, str(pocc).replace('.', '_'))

    def getWordGetter(self, lexicon_count_table=None):
        from featureGetter import FeatureGetter
        if lexicon_count_table: mm.warn(lexicon_count_table)
        wordTable = self.getWordTable() if not lexicon_count_table else lexicon_count_table

        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable), "Need to create word table to use current functionality: %s" % wordTable
        return FeatureGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host,
                             self.message_field, self.messageid_field, self.encoding,
                             self.lexicondb, featureTable=wordTable, wordTable = wordTable)
    def getWordGetterPOcc(self, pocc):
        from featureGetter import FeatureGetter
        wordTable = self.getWordTablePOcc(pocc)
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable), "Need to create word table to use current functionality"
        return FeatureGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host,
                             self.message_field, self.messageid_field, self.encoding,
                             self.lexicondb, featureTable=wordTable, wordTable = wordTable)

    def getGroupWordCounts(self, where = '', lexicon_count_table=None):
        wordGetter = self.getWordGetter(lexicon_count_table)
        return dict(wordGetter.getSumValuesByGroup(where))

    @staticmethod
    def makeBlackWhiteList(args_featlist, args_lextable, args_categories, args_lexdb):
        newlist = set()
        print "making black or white list: [%s] [%s] [%s]" %([unicode(feat,'utf-8') if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories)

        if args_lextable and args_categories:
            (conn, cur, dcur) = mm.dbConnect(args_lexdb, charset=self.encoding)
            sql = 'SELECT term FROM %s' % (args_lextable)
            if (len(args_categories) > 0) and args_categories[0] != '*':
                sql = 'SELECT term FROM %s WHERE category in (%s)'%(args_lextable, ','.join(map(lambda x: '\''+str(x)+'\'', args_categories)))

            rows = mm.executeGetList(args_lexdb, cur, sql)
            for row in rows:
                newlist.add(row[0])
        elif args_featlist:
            for feat in args_featlist:
                feat = unicode(feat, 'utf-8') if isinstance(feat, str) else feat
                # newlist.add(feat.lower())
                newlist.add(feat.upper() if sum(map(unicode.isupper, feat)) > (len(feat)/2) else feat.lower())
        else:
            raise Exception('blacklist / whitelist flag specified without providing features.')
        newlist = [w.strip() for w in newlist]
        return newlist

    ##INTERNAL METHODS##

    # def _execute(self, sql, warnMsg=True):
    #     """Executes a given query"""
    #     if warnMsg:
    #         mm.warn("SQL QUERY: %s"% sql[:fwc.MAX_SQL_PRINT_CHARS])
    #     attempts = 0;
    #     while (1):
    #         try:
    #             self.dbCursor.execute(sql)
    #             break
    #         except MySQLdb.Error, e:
    #             attempts += 1
    #             mm.warn(" *MYSQL DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
    #             time.sleep(fwc.MYSQL_ERROR_SLEEP*attempts**2)
    #             (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(self.corpdb, self.mysql_host)
    #             if (attempts > fwc.MAX_ATTEMPTS):
    #                 sys.exit(1)
    #     return True

    # def _executeGetList(self, sql, warnMsg=True):
    #     """Executes a given query, returns results as a list of lists"""
    #     if warnMsg:
    #         mm.warn("SQL QUERY: %s"% sql[:fwc.MAX_SQL_PRINT_CHARS])
    #     data = []
    #     attempts = 0;
    #     while (1):
    #         try:
    #             self.dbCursor.execute(sql)
    #             data = self.dbCursor.fetchall()
    #             break
    #         except MySQLdb.Error, e:
    #             attempts += 1
    #             mm.warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
    #             time.sleep(fwc.MYSQL_ERROR_SLEEP*attempts**2)
    #             (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(self.corpdb, self.mysql_host)
    #             if (attempts > fwc.MAX_ATTEMPTS):
    #                 sys.exit(1)
    #     return data

    # def _executeGetDict(self, sql):
    #     """Executes a given query, returns results as a list of dicts"""
    #     mm.warn("SQL (DictCursor) QUERY: %s"% sql[:fwc.MAX_SQL_PRINT_CHARS])
    #     data = []
    #     attempts = 0;
    #     while (1):
    #         try:
    #             self.dictCursor.execute(sql)
    #             data = self.dictCursor.fetchall()
    #             break
    #         except MySQLdb.Error, e:
    #             attempts += 1
    #             mm.warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
    #             time.sleep(fwc.MYSQL_ERROR_SLEEP*attempts**2)
    #             (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(self.corpdb, self.mysql_host)
    #             if (attempts > fwc.MAX_ATTEMPTS):
    #                 sys.exit(1)
    #     return data

    # def _executeGetSSCursor(self, sql, warnMsg = True):
    #     """Executes a given query (ss cursor is good to iterate over for large returns)"""
    #     if warnMsg: 
    #         mm.warn("SQL (SSCursor) QUERY: %s"% sql[:fwc.MAX_SQL_PRINT_CHARS])
    #     ssCursor = mm.dbConnect(self.corpdb, self.mysql_host)[0].cursor(MySQLdb.cursors.SSCursor)
    #     data = []
    #     attempts = 0;
    #     while (1):
    #         try:
    #             ssCursor.execute(sql)
    #             break
    #         except MySQLdb.Error, e:
    #             attempts += 1
    #             mm.warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
    #             time.sleep(fwc.MYSQL_ERROR_SLEEP*attempts**2)
    #             ssCursor = mm.dbConnect(self.corpdb, self.mysql_host)[0].cursor(MySQLdb.cursors.SSCursor)
    #             if (attempts > fwc.MAX_ATTEMPTS):
    #                 sys.exit(1)
    #     return ssCursor

    # def _executeWriteMany(self, sql, rows):
    #     """Executes a write query"""
    #     #_warn("SQL (write many) QUERY: %s"% sql)
    #     if not hasattr(self, 'writeCursor'):
    #         self.writeCursor = self.dbConn.cursor()
    #     attempts = 0;
    #     while (1):
    #         try:
    #             self.writeCursor.executemany(sql, rows)
    #             break
    #         except MySQLdb.Error, e:
    #             attempts += 1
    #             mm.warn(" *MYSQL Corpus DB ERROR on %s:\n%s (%d attempt)"% (sql, e, attempts))
    #             time.sleep(fwc.MYSQL_ERROR_SLEEP*attempts**2)
    #             (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(self.corpdb, self.mysql_host)
    #             self.writeCursor = self.dbConn.cursor()
    #             if (attempts > fwc.MAX_ATTEMPTS):
    #                 sys.exit(1)
    #     return self.writeCursor

    # ## TABLE MAINTENANCE ##

    # def _optimizeTable(self, table):
    #     """Optimizes the table -- good after a lot of deletes"""
    #     sql = """OPTIMIZE TABLE %s """%(table)
    #     return mm.execute(self.corpdb, self.dbCursor, sql) 

    # def _disableTableKeys(self, table):
    #     """Disable keys: good before doing a lot of inserts"""
    #     sql = """ALTER TABLE %s DISABLE KEYS"""%(table)
    #     return mm.execute(self.corpdb, self.dbCursor, sql) 

    # def _enableTableKeys(self, table):
    #     """Enables the keys, for use after inserting (and with keys disabled)"""
    #     sql = """ALTER TABLE %s ENABLE KEYS"""%(table)
    #     return mm.execute(self.corpdb, self.dbCursor, sql) 

    # ## Table Meta Info ##
    # def _tableExists(self, table):
    #     sql = """show tables like '%s'""" % table
    #     if mm.executeGetList(self.corpdb, self.dbCursor, sql):
    #         return True
    #     else:
    #         return False

    # def _getTableDataLength(self, table):
    #     """Returns the data length for the given table"""
    #     sql = """SELECT DATA_LENGTH FROM information_schema.tables where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'""" % (self.corpdb, table)
    #     return mm.executeGetList(self.corpdb, self.dbCursor, sql)[0]

    # def _getTableIndexLength(self, table):
    #     """Returns the data length for the given table"""
    #     sql = """SELECT INDEX_LENGTH FROM information_schema.tables where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'""" % (self.corpdb, table)
    #     return mm.executeGetList(self.corpdb, self.dbCursor, sql)[0]

    # def _getTableColumnNameTypes(self, table):
    #     """returns a dict of column names mapped to types"""
    #     sql = """SELECT column_name, column_type FROM information_schema.columns where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'"""%(self.corpdb, table)
    #     return dict(mm.executeGetList(self.corpdb, self.dbCursor, sql))


    # def _getTableColumnNameList(self, table):
    #     """returns a dict of column names mapped to types"""
    #     sql = """SELECT column_name FROM information_schema.columns where TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s' ORDER BY ORDINAL_POSITION"""%(self.corpdb, table)
    #     return [x[0] for x in mm.executeGetList(self.corpdb, self.dbCursor, sql)]

    


# at bottom to avoid circular imports
#from featureGetter import FeatureGetter
#################################################################
### Main / Command-Line Processing:
##
#
if __name__ == "__main__":
    ##Argument Parser:
    print "featureWorker.py Command-line Interface is Deprecated.\nUse fwInterface.py"
    sys.exit(0)


        
        
########################################################################################
### Saved Code
##
#

# from sklearn.linear_model import LinearRegression
#        lr = None
#        if controls: lr = LinearRegression(normalize=True) #TODO: drop normalize = true if using feat_norm
#                        (X, y) = alignDictsAsXy(X = controls.append(dataDict), y = outcomes)
#                        lr.fit(X, y)
#                        tup = (lr.coef_[-1], )


        # for i in range(0, len(words)):
        #     for j in range(0, min(n, len(words)-i)):
        #         gram = ' '.join(words[i:(i+j+1)])
        #         if not gram in freqs:
        #             freqs[gram] = 1
        #         else: 
        #             freqs[gram] += 1
#
#now using rank data (better to rank at the list/array level rather than dict
#def floatDictToRankDict(d):
#    """Converts a dictionary with float values to rank values"""
#    sortedValues = sorted(d.values())
#    valueToRankItems = [(sortedValues[i], i) for i in xrange(len(sortedValues))]
#    valueToRank = dict(valueToRankItems)
#    #handle ties: TODO            
#    newD = dict()
#    for k, v in d.iteritems():
#        newD[k] = valueToRank[v]
#    return newD

#    #check and remove NaNs: (comment line above to add back in)
#    newX, newy = list(), list()
#    for row in xrange(len(listy)):
#        removeRow = False
#        if isnan(listy[row]): removeRow = True
#        else: 
#            for col in range(len(listX[row])):
#                if isnan(listX[row][col]): 
#                    removeRow = True
#                    break
#        if not removeRow:
#            newX.append(listX[row])
#            newy.append(listy[row])
#    return (newX, newy)  

        # if perc < 0.25: #light-grey to grey (average 160)
        #     (red, green, blue) = rgbColorMix((212, 212, 160), (160, 160, 80), resolution)[int(((1.00-(1-perc))/0.25)*resolution) - 1]
        # elif perc >= 0.25 and perc < 0.50: #grey to green (average 128)
        #     (red, green, blue) = rgbColorMix((160, 160, 80), (68, 168, 68), resolution)[int(((0.75-(1-perc))/0.25)*resolution) - 1]
        # elif perc >= 0.50 and perc < 0.75: #green to blue (average 92)
        #     (red, green, blue) = rgbColorMix((68, 168, 68), (40, 40, 128), resolution)[int(((0.50-(1-perc))/0.25)*resolution) - 1]
        # else: #blue to red (average 64)
        #     (red, green, blue) = rgbColorMix((40, 40, 128), (96, 0, 0), resolution)[int(((0.25-(1-perc))/0.25)*resolution) - 1]

        ##rainbow
        # if perc < 0.33: #yellow to green (128)
        #     (red, green, blue) = rgbColorMix((208, 208, 112), (96, 192, 96), resolution)[int(((1.00-(1-perc))/0.33)*resolution) - 1]
        # elif perc >= 0.33 and perc < 0.66: #green (128) to blue (80)
        #     (red, green, blue) = rgbColorMix((96, 192, 96), (56, 56, 128), resolution)[int(((0.66-(1-perc))/0.33)*resolution) - 1]
        # else: #blue (80) to red(32)
        #     (red, green, blue) = rgbColorMix((56, 56, 128), (96, 0, 0), resolution)[int(((0.33-(1-perc))/0.33)*resolution) - 1]

        #sonar basic
        # if perc < 0.333: #blue to green
        #     (red, green, blue) = rgbColorMix((12, 12, 236),(12, 236, 12), resolution)[int(((1.00-(1-perc))/0.333)*resolution) - 1]
        # elif perc >= 0.333 and perc < 0.666: #green to yellow
        #     (red, green, blue) = rgbColorMix((12, 236, 12), (200, 200, 12), resolution)[int(((0.666-(1-perc))/0.333)*resolution) - 1]
        # else: #red to black
        #     (red, green, blue) = rgbColorMix((200, 200, 12), (236, 12, 12), resolution)[int(((0.333-(1-perc))/0.333)*resolution) - 1]


        #frosty to red
        # if perc < 0.30: #teal to blue
        #     (red, green, blue) = rgbColorMix((112, 208, 208), (96, 104, 184), resolution)[int(((1.00-(1-perc))/0.3)*resolution) - 1]
        # elif perc >= 0.30 and perc < 0.60: #green (128) to blue (80)
        #     (red, green, blue) = rgbColorMix((96, 104, 184), (52, 44, 144), resolution)[int(((0.7-(1-perc))/0.3)*resolution) - 1]
        # else: #blue (80) to red(32)
        #     (red, green, blue) = rgbColorMix((52, 44, 144), (96, 0, 0), resolution)[int(((0.4-(1-perc))/0.4)*resolution) - 1]

    ##heat:
        # if perc < 0.30: #yellow to orange
        #     (red, green, blue) = rgbColorMix((236, 236, 64), (232, 120, 32), re