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
    def __init__(self, corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb = fwc.DEF_LEXICON_DB, date_field=fwc.DEF_DATE_FIELD, wordTable = None):
        self.corpdb = corpdb
        self.corptable = corptable
        self.correl_field = correl_field
        self.mysql_host = mysql_host
        self.message_field = message_field
        self.messageid_field = messageid_field
        self.encoding = encoding
        self.use_unicode = use_unicode
        (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(corpdb, host=mysql_host, charset=encoding, use_unicode=self.use_unicode)
        self.lexicondb = lexicondb
        self.wordTable = wordTable if wordTable else "feat$1gram$%s$%s$16to16"%(self.corptable, self.correl_field)

    ##PUBLIC METHODS#
    def getMessages(self, messageTable = None, where = None):
        """..."""
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s FROM %s"""% (self.messageid_field, self.message_field, messageTable)
        if where: msql += " WHERE " + where
        return mm.executeGetSSCursor(self.corpdb, msql, charset=self.encoding, use_unicode=self.use_unicode, host=self.mysql_host)

    def getMessagesForCorrelField(self, cf_id, messageTable = None, warnMsg = True):
        """..."""
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s FROM %s WHERE %s = '%s'""" % (
            self.messageid_field, self.message_field, messageTable, self.correl_field, cf_id)
        #return self._executeGetSSCursor(msql, warnMsg, host=self.mysql_host)
        return mm.executeGetList(self.corpdb, self.dbCursor, msql, warnMsg, charset=self.encoding, use_unicode=self.use_unicode)

    def getMessagesWithFieldForCorrelField(self, cf_id, extraField, messageTable = None, warnMsg = True):
        """..."""
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s, %s FROM %s WHERE %s = '%s'""" % (
            self.messageid_field, self.message_field, extraField, messageTable, self.correl_field, cf_id)
        #return self._executeGetSSCursor(msql, showQuery)
        return mm.executeGetList(self.corpdb, self.dbCursor, msql, warnMsg, charset=self.encoding, use_unicode=self.use_unicode)

    def getNumWordsByCorrelField(self, where = ''):
        """..."""
        #assumes corptable has num_words field for each message
        #SELECT user_id, sum(num_words) FROM (SELECT user_id, num_words FROM messages GROUP BY message_id) as a GROUP BY user_id
        sql = """SELECT %s, sum(num_words) FROM (SELECT %s, num_words FROM %s """ % (self.correl_field, self.correl_field, self.corptable)
        if (where): sql += ' WHERE ' + where  
        sql += """ GROUP BY %s) as a """ % self.messageid_field 
        sql += """ GROUP BY %s """ % self.correl_field
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

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
        if lexicon_count_table: fwc.warn(lexicon_count_table)
        wordTable = self.getWordTable() if not lexicon_count_table else lexicon_count_table

        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable), "Need to create word table to use current functionality: %s" % wordTable
        return FeatureGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host,
                             self.message_field, self.messageid_field, self.encoding, self.use_unicode, 
                             self.lexicondb, featureTable=wordTable, wordTable = wordTable)
    def getWordGetterPOcc(self, pocc):
        from featureGetter import FeatureGetter
        wordTable = self.getWordTablePOcc(pocc)
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode), "Need to create word table to use current functionality"
        return FeatureGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host,
                             self.message_field, self.messageid_field, self.encoding, self.use_unicode, 
                             self.lexicondb, featureTable=wordTable, wordTable = wordTable)

    def getGroupWordCounts(self, where = '', lexicon_count_table=None):
        wordGetter = self.getWordGetter(lexicon_count_table)
        return dict(wordGetter.getSumValuesByGroup(where))

    def getFeatureTables(self, where = ''):
        """Return all available feature tables for the given corpdb, corptable and correl_field"""
        sql = """SHOW TABLES FROM %s LIKE 'feat$%%$%s$%s$%%' """ % (self.corpdb, self.corptable, self.correl_field)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    @staticmethod
    def makeBlackWhiteList(args_featlist, args_lextable, args_categories, args_lexdb, args_use_unicode):
        newlist = set()
        if args_use_unicode:
            print "making black or white list: [%s] [%s] [%s]" %([unicode(feat,'utf-8') if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories)
        else:
            print "making black or white list: [%s] [%s] [%s]" %([feat if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories)
        if args_lextable and args_categories:
            (conn, cur, dcur) = mm.dbConnect(args_lexdb, charset=self.encoding, use_unicode=self.use_unicode)
            sql = 'SELECT term FROM %s' % (args_lextable)
            if (len(args_categories) > 0) and args_categories[0] != '*':
                sql = 'SELECT term FROM %s WHERE category in (%s)'%(args_lextable, ','.join(map(lambda x: '\''+str(x)+'\'', args_categories)))

            rows = mm.executeGetList(args_lexdb, cur, sql, charset=self.encoding, use_unicode=self.use_unicode)
            for row in rows:
                newlist.add(row[0])
        elif args_featlist:
            for feat in args_featlist:
                if args_use_unicode:
                    feat = unicode(feat, 'utf-8') if isinstance(feat, str) else feat
                else:
                    feat = feat if isinstance(feat, str) else feat
                # newlist.add(feat.lower())
                if args_use_unicode:
                    newlist.add(feat.upper() if sum(map(unicode.isupper, feat)) > (len(feat)/2) else feat.lower())
                else:
                    newlist.add(feat.upper() if sum(map(str.isupper, feat)) > (len(feat)/2) else feat.lower())
        else:
            raise Exception('blacklist / whitelist flag specified without providing features.')
        newlist = [w.strip() for w in newlist]
        return newlist

#################################################################
### Main / Command-Line Processing:
##
#
if __name__ == "__main__":
    ##Argument Parser:
    print "featureWorker.py Command-line Interface is Deprecated.\nUse fwInterface.py"
    sys.exit(0)

