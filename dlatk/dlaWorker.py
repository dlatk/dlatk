import sys
import time
import MySQLdb

from . import dlaConstants as dlac
from .mysqlmethods import mysqlMethods as mm 

class DLAWorker(object):
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
    DLAWorker object
    """
    
    def __init__(self, corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb = dlac.DEF_LEXICON_DB, date_field=dlac.DEF_DATE_FIELD, wordTable = None):
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
    def checkIndices(self, table, primary=False, correlField=False):
        hasPrimary, hasCorrelIndex = True, True
        warn_message = "WARNING: The table %s does not have:"  % table
        if primary:
            hasPrimary = mm.primaryKeyExists(self.dbConn, self.dbCursor, table, correlField)
            if not hasPrimary: warn_message += " a PRIMARY key on %s" % correlField
        if correlField:
            hasCorrelIndex = mm.indexExists(self.dbConn, self.dbCursor, table, correlField)
            if not hasCorrelIndex: 
                if not hasPrimary: warn_message += " or"
                warn_message += " an index on %s" % correlField
        warn_message += ". Consider adding."
        if not hasPrimary or not hasCorrelIndex:
            dlac.warn(warn_message)

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
        if lexicon_count_table: dlac.warn(lexicon_count_table)
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

    def getTables(self, feat_table = False, like = None):
        """Returns a list of available tables.
 
        Parameters
        ----------
        feat_table : :obj:`str`, optional
            Indicator for listing feature tables

        like : :obj:`boolean`, optional
            Filter tables with sql-style call.
     
        Returns
        -------
        list
            A list of tables names
        """
        if feat_table:
            sql = """SHOW TABLES FROM %s LIKE 'feat$%%$%s$%s$%%' """ % (self.corpdb, self.corptable, self.correl_field)
        else:
            sql = """SHOW TABLES FROM %s where Tables_in_%s NOT LIKE 'feat%%' """ % (self.corpdb, self.corpdb)
            if isinstance(like, str): sql += """ AND Tables_in_%s like '%s'""" % (self.corpdb, like)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    def describeTable(self, table_name):
        """
 
        Parameters
        ----------
        table_name : :obj:`str`
            Name of table to describe
     
        Returns
        -------
        Description of table (list of lists)
            
        """
        sql = """DESCRIBE %s""" % (table_name)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    def viewTable(self, table_name):
        """
 
        Parameters
        ----------
        table_name : :obj:`str`
            Name of table to describe
     
        Returns
        -------
        First 5 rows of table (list of lists)
            
        """
        col_sql = """select column_name from information_schema.columns 
            where table_schema = '%s' and table_name='%s'""" % (self.corpdb, table_name)
        col_names = [col[0] for col in mm.executeGetList(self.corpdb, self.dbCursor, col_sql, charset=self.encoding, use_unicode=self.use_unicode)]
        sql = """SELECT * FROM %s LIMIT 10""" % (table_name)
        return [col_names] + list(mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode))


    def createRandomSample(self, percentage, random_seed = dlac.DEFAULT_RANDOM_SEED, where = ''):
        """Creates a new table from a random subetset of rows.
 
        Parameters
        ----------
        percentage : :obj:`float`, optional
            Percentage of random rows to keep

        random_seed : :obj:`int`, optional
            Filter groups with sql-style call.
     
        Returns
        -------
        list
            A list of feature tables names
        """
        new_table = self.corptable + "_rand"

        rows_sql = """SELECT count(*) from %s""" % (self.corptable)
        n_old_rows = mm.executeGetList1(self.corpdb, self.dbCursor, rows_sql, charset=self.encoding, use_unicode=self.use_unicode)[0]
        n_new_rows = round(percentage*n_old_rows)

        drop_sql = """DROP TABLE IF EXISTS %s""" % (new_table)
        mm.execute(self.corpdb, self.dbCursor, drop_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)

        create_sql = """CREATE TABLE %s LIKE %s""" % (new_table, self.corptable)
        mm.execute(self.corpdb, self.dbCursor, create_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)
        
        disable_sql = """ALTER TABLE %s DISABLE KEYS""" % (new_table)
        mm.execute(self.corpdb, self.dbCursor, disable_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)
        
        insert_sql = """INSERT INTO %s SELECT * FROM %s where RAND(%s) < %s""" % (new_table, self.corptable, random_seed, percentage*1.1)
        if where: insert_sql += " AND %s" % (where)
        insert_sql += " LIMIT %s" % (n_new_rows)
        mm.execute(self.corpdb, self.dbCursor, insert_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)
        
        enable_sql = """ALTER TABLE %s ENABLE KEYS""" % (new_table)
        mm.execute(self.corpdb, self.dbCursor, enable_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)

        return new_table

    def createCopiedTable(self, old_table, new_table, where = ''):
        """Creates a new table as a copy of an old table.
 
        Parameters
        ----------
        old_table: :obj:`string`, 
            name of the table to be copied

        new_table: :obj:`string`, 
            name of the new table

        Returns
        -------
        string
            new table name
        """
        #drop_sql = """DROP TABLE IF EXISTS %s""" % (new_table)
        #mm.execute(self.corpdb, self.dbCursor, drop_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)

        create_sql = """CREATE TABLE %s LIKE %s""" % (new_table, old_table)
        mm.execute(self.corpdb, self.dbCursor, create_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)
        
        disable_sql = """ALTER TABLE %s DISABLE KEYS""" % (new_table)
        mm.execute(self.corpdb, self.dbCursor, disable_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)
        
        insert_sql = """INSERT INTO %s SELECT * FROM %s""" % (new_table, old_table)
        if where: insert_sql += " WHERE %s" % (where)
        mm.execute(self.corpdb, self.dbCursor, insert_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)
        
        enable_sql = """ALTER TABLE %s ENABLE KEYS""" % (new_table)
        mm.execute(self.corpdb, self.dbCursor, enable_sql, warnQuery=True, charset=self.encoding, use_unicode=self.use_unicode)

        return new_table

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
            print("making black or white list: [%s] [%s] [%s]" %([feat if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories))
        else:
            print("making black or white list: [%s] [%s] [%s]" %([feat if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories))
        if args_lextable:
            (conn, cur, dcur) = mm.dbConnect(args_lexdb, charset=dlac.DEF_ENCODING, use_unicode=args_use_unicode)
            sql = 'SELECT term FROM %s' % (args_lextable)
            if (len(args_categories) > 0) and args_categories[0] != '*':
                sql = 'SELECT term FROM %s WHERE category in (%s)'%(args_lextable, ','.join(['\''+str(x)+'\'' for x in args_categories]))

            rows = mm.executeGetList(args_lexdb, cur, sql, charset=dlac.DEF_ENCODING, use_unicode=args_use_unicode)
            for row in rows:
                newlist.add(row[0])
        elif args_featlist:
            for feat in args_featlist:
                if args_use_unicode:
                    feat = feat if isinstance(feat, str) else feat
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
    print("dlaWorker.py Command-line Interface is Deprecated.\nUse dlatkInterface.py")
    sys.exit(0)

