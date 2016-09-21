import sys
import time
import MySQLdb

from . import fwConstants as fwc
from .mysqlMethods import mysqlMethods as mm 

class FeatureWorker(object):
    """Generic class for functions working with features

    Parameters
    ----------
    corpdb : str
        Corpus Database Name.
    corptable : str
        Corpus Table.
    correl_field : str
        Correlation Field (AKA Group Field): The field which features are aggregated over.
    mysql_host : str
        Host that the mysql server runs on.
    message_field : str
        The field where the text to be analyzed is located.
    messageid_field : str
        The unique identifier for the message.
    encoding : str
        MySQL encoding
    lexicondb : :obj:`str`, optional
        The database which stores all lexicons.
    date_field : :obj:`str`, optional
        Date a message was sent (if avail, for timex processing).
    wordTable : :obj:`str`, optional
        Table that contains the list of words to give for lex extraction/group_freq_thresh

    Returns
    -------
    FeatureWorker object
    """
    
    def __init__(self, corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb = fwc.DEF_LEXICON_DB, date_field=fwc.DEF_DATE_FIELD, wordTable = None):
        self.corpdb = corpdb
        self.corptable = corptable
        self.correl_field = correl_field
        self.mysql_host = mysql_host
        self.message_field = message_field
        self.messageid_field = messageid_field
        self.encoding = encoding
        self.use_unicode = use_unicode
        (self.dbConn, self.dbCursor, self.dictCursor) = mm.dbConnect(corpdb, host=mysql_host, charset=encoding)
        self.lexicondb = lexicondb
        self.wordTable = wordTable if wordTable else "feat$1gram$%s$%s$16to16"%(self.corptable, self.correl_field)

    ##PUBLIC METHODS#
    def getMessages(self, messageTable = None, where = None):
        """?????
 
        Parameters
        ----------
        messageTable : :obj:`str`, optional
            Name of message table.
        where : :obj:`str`, optional
            Filter groups with sql-style call.
     
        Returns
        -------
        ?????
            ?????
        """
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s FROM %s"""% (self.messageid_field, self.message_field, messageTable)
        if where: msql += " WHERE " + where
        return mm.executeGetSSCursor(self.corpdb, msql, charset=self.encoding, host=self.mysql_host)

    def getMessagesForCorrelField(self, cf_id, messageTable = None, warnMsg = True):
        """?????
 
        Parameters
        ----------
        cf_id : str
            Correl field id.
        messageTable : :obj:`str`, optional
            name of message table.
        warnMsg : :obj:`boolean`, optional
            ?????
     
        Returns
        -------
        ?????
            ?????
        """
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s FROM %s WHERE %s = '%s'""" % (
            self.messageid_field, self.message_field, messageTable, self.correl_field, cf_id)
        #return self._executeGetSSCursor(msql, warnMsg, host=self.mysql_host)
        return mm.executeGetList(self.corpdb, self.dbCursor, msql, warnMsg, charset=self.encoding)

    def getMessagesWithFieldForCorrelField(self, cf_id, extraField, messageTable = None, warnMsg = True):
        """?????
 
        Parameters
        ----------
        cf_id : str
            Correl field id.
        extraField : str
            ?????
        messageTable : :obj:`str`, optional
            name of message table.
        warnMsg : :obj:`boolean`, optional
            ?????
     
        Returns
        -------
        describe : list
            A list of messages for a given correl field id.
        """
        if not messageTable: messageTable = self.corptable
        msql = """SELECT %s, %s, %s FROM %s WHERE %s = '%s'""" % (
            self.messageid_field, self.message_field, extraField, messageTable, self.correl_field, cf_id)
        #return self._executeGetSSCursor(msql, showQuery)
        return mm.executeGetList(self.corpdb, self.dbCursor, msql, warnMsg, charset=self.encoding)

    def getNumWordsByCorrelField(self, where = ''):
        """?????
 
        Parameters
        ----------
        where : :obj:`str`, optional
            Filter groups with sql-style call.
     
        Returns
        -------
        ?????
            ?????
        """
        #assumes corptable has num_words field for each message
        #SELECT user_id, sum(num_words) FROM (SELECT user_id, num_words FROM messages GROUP BY message_id) as a GROUP BY user_id
        sql = """SELECT %s, sum(num_words) FROM (SELECT %s, num_words FROM %s """ % (self.correl_field, self.correl_field, self.corptable)
        if (where): sql += ' WHERE ' + where  
        sql += """ GROUP BY %s) as a """ % self.messageid_field 
        sql += """ GROUP BY %s """ % self.correl_field
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    def getWordTable(self, corptable = None):
        """?????
 
        Parameters
        ----------
        corptable : :obj:`str`, optional
            Choices in brackets, default first when optional.
     
        Returns
        -------
        str
            Name of word table for given corptable and correl_field.
        """
        if self.wordTable: return self.wordTable
        if not corptable:
            corptable  = self.corptable
        return "feat$1gram$%s$%s$16to16"%(corptable, self.correl_field)

    def get1gramTable(self):
        """?????
     
        Returns
        -------
        str
            Name of word table (1gram table) for a given corptable and correl field.
        """
        return "feat$1gram$%s$%s$16to16"%(self.corptable, self.correl_field)
        
    def getWordTablePOcc(self, pocc):
        """This function does something.
 
        Parameters
        ----------
        pocc : float
            Value of p occurrence filter.
     
        Returns
        -------
        str
            A word table (1gram table) for a given corptable, correl field and p occurrence value.
        """
        return "feat$1gram$%s$%s$16to16$%s"%(self.corptable, self.correl_field, str(pocc).replace('.', '_'))

    def getWordGetter(self, lexicon_count_table=None):
        """Returns a FeatureGetter used for getting word counts. Usually used for group_freq_thresh.
 
        Parameters
        ----------
        lexicon_count_table : :obj:`str`, optional
            name of message table.
     
        Returns
        -------
        FeatureGetter
        """
        from .featureGetter import FeatureGetter
        if lexicon_count_table: fwc.warn(lexicon_count_table)
        wordTable = self.getWordTable() if not lexicon_count_table else lexicon_count_table

        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable), "Need to create word table to use current functionality: %s" % wordTable
        return FeatureGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host,
                             self.message_field, self.messageid_field, self.encoding, self.use_unicode, 
                             self.lexicondb, featureTable=wordTable, wordTable = wordTable)
    
    def getWordGetterPOcc(self, pocc):
        """Returns a FeatureGetter for given p_occ filter values used for getting word counts. Usually used for group_freq_thresh.
 
        Parameters
        ----------
        pocc : float
            Value of p occurrence filter.
     
        Returns
        -------
        FeatureGetter
        """
        from .featureGetter import FeatureGetter
        wordTable = self.getWordTablePOcc(pocc)
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode), "Need to create word table to use current functionality"
        return FeatureGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host,
                             self.message_field, self.messageid_field, self.encoding, self.use_unicode, 
                             self.lexicondb, featureTable=wordTable, wordTable = wordTable)

    def getGroupWordCounts(self, where = '', lexicon_count_table=None):
        """Get word counts for groups
 
        Parameters
        ----------
        where : :obj:`str`, optional
            Filter groups with sql-style call.
        lexicon_count_table : :obj:`str`, optional
            ?????
     
        Returns
        -------
        dict
            {group_id: sum(values)}
        """
        wordGetter = self.getWordGetter(lexicon_count_table)
        return dict(wordGetter.getSumValuesByGroup(where))

    def getFeatureTables(self, where = ''):
        """Returns a list of available feature tables for a given corptable and correl_field.
 
        Parameters
        ----------
        where : :obj:`str`, optional
            Filter groups with sql-style call.
     
        Returns
        -------
        list
            A list of feature tables names
        """
        sql = """SHOW TABLES FROM %s LIKE 'feat$%%$%s$%s$%%' """ % (self.corpdb, self.corptable, self.correl_field)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    @staticmethod
    def makeBlackWhiteList(args_featlist, args_lextable, args_categories, args_lexdb, args_use_unicode):
        """?????
 
        Parameters
        ----------
        args_featlist : str
            ?????
        args_lextable : str
            ?????
        args_categories : str
            ?????
        args_lexdb : str
            ?????
        args_use_unicode : boolean
            ?????
     
        Returns
        -------
        newlist : list
            ?????
        """
        newlist = set()
        if args_use_unicode:
            print("making black or white list: [%s] [%s] [%s]" %([str(feat,'utf-8') if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories))
        else:
            print("making black or white list: [%s] [%s] [%s]" %([feat if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories))
        if args_lextable and args_categories:
            (conn, cur, dcur) = mm.dbConnect(args_lexdb, charset=self.encoding, use_unicode=self.use_unicode)
            sql = 'SELECT term FROM %s' % (args_lextable)
            if (len(args_categories) > 0) and args_categories[0] != '*':
                sql = 'SELECT term FROM %s WHERE category in (%s)'%(args_lextable, ','.join(['\''+str(x)+'\'' for x in args_categories]))

            rows = mm.executeGetList(args_lexdb, cur, sql, charset=self.encoding, use_unicode=self.use_unicode)
            for row in rows:
                newlist.add(row[0])
        elif args_featlist:
            for feat in args_featlist:
                if args_use_unicode:
                    feat = str(feat, 'utf-8') if isinstance(feat, str) else feat
                else:
                    feat = feat if isinstance(feat, str) else feat
                # newlist.add(feat.lower())
                if args_use_unicode:
                    newlist.add(feat.upper() if sum(map(str.isupper, feat)) > (len(feat)/2) else feat.lower())
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
    print("featureWorker.py Command-line Interface is Deprecated.\nUse dlatkInterface.py")
    sys.exit(0)

