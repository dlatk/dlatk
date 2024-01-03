from os import path, makedirs
import sys
import time
import csv
try:
    import MySQLdb
except:
    pass

from .database.dataEngine import DataEngine
from .database.query import QueryBuilder
from .lexicainterface.lexInterface import WeightedLexicon, loadWeightedLexiconFromSparse, loadWeightedLexiconFromTopicCSV

from . import dlaConstants as dlac
from .mysqlmethods import mysqlMethods as mm 

class DLAWorker(object):
    """Generic class for functions working with features

    Key Parameters
    --------------
    corpdb : str
        Corpus Database Name.
    corptable : str
        Corpus Table (AKA message_table).
    correl_field : str
        Correlation Field (AKA Group Field): The field which features are aggregated over.

    Advanced Parameters
    -------------------
    data_engine: object (updated outside constructor)
        the engine for working with data. Currently supports: mysql, sqlite
    mysql_config_file : str
        path to MySQL config file, for example ~/.my.cnf
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
    
    def __init__(self, db_type, corpdb, corptable, correl_field, mysql_config_file, message_field, messageid_field, encoding, use_unicode, lexicondb = dlac.DEF_LEXICON_DB, date_field=dlac.DEF_DATE_FIELD, wordTable=None):
        self.corpdb = corpdb
        self.corptable = corptable
        self.db_type = db_type.lower()
        self.encoding = encoding
        self.use_unicode = use_unicode
        self.mysql_config_file = mysql_config_file

        self.prepare_corpdb()

        self.correl_field = correl_field
        self.message_field = message_field
        self.messageid_field = messageid_field

        self.qb = QueryBuilder(self.data_engine)

        self.lexicondb = lexicondb
        self.lexicon = None

        if wordTable:
            self.wordTable = wordTable
        else:
            wordTable = "feat$1gram$%s$%s"%(self.corptable, self.correl_field)
            if not self.data_engine.tableExists(wordTable):
                wordTable = "feat$1gram$%s$%s"%(self.corptable, self.correl_field)
            self.wordTable = wordTable
        self.messageIdUniqueChecked = False
    
    def prepare_corpdb(self):

        if (".csv" in self.corptable) or (self.db_type == "sqlite"):

            self.db_type = "sqlite"

            default_dir = path.join("/content", "sqlite_data") if path.exists("/content") else path.join(path.expanduser('~'), "sqlite_data")
            if not path.exists(default_dir):
                makedirs(default_dir)

            if self.corpdb is None:
                self.corpdb = self.corptable.split('/')[-1].split('.')[0]
            self.corpdb = path.join(default_dir, self.corpdb)
            
            print("Connecting to SQLite database: {}.db".format(self.corpdb))
            self.data_engine = DataEngine(self.corpdb, self.mysql_config_file, self.encoding, self.use_unicode, self.db_type)
            (self.dbConn, self.dbCursor, self.dictCursor) = self.data_engine.connect()

            message_table = self.corptable.split('/')[-1].split('.')[0]
            if not self.data_engine.tableExists(message_table):
                if ".csv" in self.corptable:
                    self.data_engine.csvToTable(self.corptable, message_table)
                else:
                    dlac.warn("Message table missing")
            
            self.corptable = message_table

        elif self.db_type == "mysql":

            self.data_engine = DataEngine(self.corpdb, self.mysql_config_file, self.encoding, self.use_unicode, self.db_type)
            (self.dbConn, self.dbCursor, self.dictCursor) = self.data_engine.connect()

            if not self.data_engine.tableExists(self.corptable):
                print("Message table missing")
                sys.exit(1)

    def load_lexicon(self, lexicon, lexicon_type="sparse", table_name=None):

        lex_to_func = {
            "sparse": loadWeightedLexiconFromSparse,
            "topicCSV": loadWeightedLexiconFromTopicCSV}

        idx_to_db_type = ["sqlite", "mysql"]
        db_type_to_idx = {db_type: index for index, db_type in enumerate(idx_to_db_type)}
        db_type = self.db_type
        db_idx = db_type_to_idx[db_type]
        db_name = self.lexicondb

        if db_type == "sqlite":
            default_dir = path.join("/content", "sqlite_data") if path.exists("/content") else path.join(path.expanduser('~'), "sqlite_data")
            self.lexicondb = path.join(default_dir, db_name)
        if db_type == "mysql":
            self.lexicondb = db_name

        self.lexicon = WeightedLexicon(
            lexicon_db=self.lexicondb,
            mysql_config_file=self.mysql_config_file,
            lex_db_type=db_type,
            encoding=self.encoding,
            use_unicode=self.use_unicode)

        lex_table = lexicon.split('/')[-1].split('.')[0] if table_name is None else table_name
        if not self.lexicon.engine.tableExists(lex_table):

            if ".csv" in lexicon:
                self.lexicon.setWeightedLexicon(lex_to_func[lexicon_type](lexicon))
                self.lexicon.createLexiconTable(lex_table)

            else:
                #rotate connection if lexicon not in current db type
                db_idx = (db_idx + 1) % len(idx_to_db_type)
                db_type = idx_to_db_type[db_idx]

                if db_type == "sqlite":
                    default_dir = path.join("/content", "sqlite_data") if path.exists("/content") else path.join(path.expanduser('~'), "sqlite_data")
                    self.lexicondb = path.join(default_dir, db_name)
                if db_type == "mysql":
                    self.lexicondb = db_name

                self.lexicon = WeightedLexicon(
                    lexicon_db=self.lexicondb,
                    mysql_config_file=self.mysql_config_file, 
                    lex_db_type=db_type, 
                    encoding=self.encoding, 
                    use_unicode=self.use_unicode)

        return lex_table
 
    ##PUBLIC METHODS#
    def checkIndices(self, table, primary=False, correlField=False):
        hasPrimary, hasCorrelIndex = True, True
        warn_message = "WARNING: The table %s does not have:"  % table
        if primary:
            hasPrimary = self.data_engine.primaryKeyExists(table, correlField) 
            if not hasPrimary: warn_message += " a PRIMARY key on %s" % correlField
        elif correlField:
            hasCorrelIndex = self.data_engine.indexExists(table, correlField) 

            if not hasCorrelIndex: 
                if not hasPrimary: warn_message += " or"
                warn_message += " an index on %s" % correlField
        warn_message += ". Consider adding."
        if self.messageid_field == correlField and not hasPrimary:
            warn_message += "\n         Please check that all messages have a unique %s, this can significantly impact all downstream analysis" % (self.messageid_field)
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
        selectQuery = self.qb.create_select_query(messageTable).set_fields([self.messageid_field, self.message_field])
        if where: selectQuery = selectQuery.where(where)
        
        return selectQuery.execute_query()

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
        # check that self.messageid_field is unique
        if self.messageIdUniqueChecked == False:
            self.checkIndices(messageTable, primary=True, correlField=self.messageid_field)
            if self.correl_field != self.messageid_field:
                self.checkIndices(messageTable, primary=False, correlField=self.correl_field)
            self.messageIdUniqueChecked = True
        where_conditions = """%s='%s'"""%(self.correl_field, cf_id)
        selectQuery = self.qb.create_select_query(messageTable).set_fields([self.messageid_field, self.message_field]).where(where_conditions)
        return selectQuery.execute_query()

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
        fields = [self.messageid_field, self.message_field, extraField]
        where_condition = "%s='%s'"%(self.correl_field, cf_id)

        selectQuery = self.qb.create_select_query(messageTable).where(where_condition).set_fields(fields)
        
        return selectQuery.execute_query()

    def getMidAndExtraForCorrelField(self, cf_id, extraField, messageTable = None, where = None, warnMsg = True):
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
        fields = [self.messageid_field, extraField]
        where_condition = "%s='%s'"%(self.correl_field, cf_id)
        if where: where_condition + " AND %s"%(where)

        selectQuery = self.qb.create_select_query(messageTable).where(where_condition).set_fields(fields)

        return selectQuery.execute_query()
    
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
        fields = [self.correl_field, "num_words"]
        where_condition = where if where else ''
        group_by_fields = [self.messageid_field] 
        intermediate_query = self.qb.create_select_query(self.corptable).where(where_condition).group_by(group_by_fields).set_fields(fields).toString()
        
        fields = [self.correl_field, "SUM(num_words)"]
        table_name = intermediate_query + "AS a" #alias before select
        group_by_fields = [self.correl_field]

        selectQuery = self.qb.create_select_query(table_name).group_by(group_by_fields).set_fields(fields)
        return selectQuery.execute_query()

    def getWordTable(self, corptable = None):
        """returns the table to use for word counts
 
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
        
        wordTable = "feat$1gram$%s$%s"%(corptable, self.correl_field)
        if not self.data_engine.tableExists(wordTable):
            wordTable = "feat$1gram$%s$%s"%(corptable, self.correl_field)
        return wordTable

    def get1gramTable(self):
        """?????
     
        Returns
        -------
        str
            Name of word table (1gram table) for a given corptable and correl field.
        """
        wordTable = "feat$1gram$%s$%s"%(self.corptable, self.correl_field)
        if not self.data_engine.tableExists(wordTable):
            wordTable = "feat$1gram$%s$%s"%(self.corptable, self.correl_field)
        return wordTable
        
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
        wordTable = "feat$1gram$%s$%s$%s"%(self.corptable, self.correl_field, str(pocc).replace('.', '_'))
        if not self.data_engine.tableExists(wordTable):
            wordTable = "feat$1gram$%s$%s$%s"%(self.corptable, self.correl_field, str(pocc).replace('.', '_'))
        return wordTable

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

        assert self.data_engine.tableExists(wordTable), "Need to create word table to use current functionality: %s" % wordTable
        return FeatureGetter(self.db_type, self.corpdb, self.corptable, self.correl_field, self.mysql_config_file, self.message_field, self.messageid_field, self.encoding, self.use_unicode, self.lexicondb, featureTable=wordTable, wordTable=wordTable)
    
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
        
        assert self.data_engine.tableExists(wordTable), "Need to create word table to use current functionality"
        return FeatureGetter(self.db_type, self.corpdb, self.corptable, self.correl_field, self.mysql_config_file,
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
            like = "feat$%${}${}%".format(self.corptable, self.correl_field)
            return self.data_engine.getTables(like, feat_table)
            
        return self.data_engine.getTables(like)
        
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
        dlac.warn("\nPrinting: {}\n----\n".format(table_name))
        return self.data_engine.describeTable(table_name)

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
        dlac.warn("\nPrinting: {}\n----\n".format(table_name))
        return self.data_engine.viewTable(table_name)


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
        
        fields = ["count(*)"]
        selectQuery = self.qb.create_select_query(self.corptable).set_fields(fields)
        n_old_rows = selectQuery.execute_query()[0][0]
        n_new_rows = round(percentage * n_old_rows)

        dropQuery = self.qb.create_drop_query(new_table)
        dropQuery.execute_query()

        createQuery = self.qb.create_createTable_query(new_table).like(self.corptable)
        createQuery.execute_query()
        
        self.data_engine.disable_table_keys(new_table)

        where_condition = "WHERE {} < {}".format(self.data_engine.getRandomFunc(random_seed), percentage * 1.1)
        if where: where_condition += " AND {}".format(where)
        where_condition += " LIMIT {}".format(n_new_rows)
        selectQuery = self.qb.create_select_query(self.corptable).set_fields(['*']).where(where_condition)
        insertQuery = self.qb.create_insert_query(new_table).values_from_select(selectQuery)
        insertQuery.execute_query()
        
        self.data_engine.enable_table_keys(new_table)

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

        dropQuery = self.qb.create_drop_query(new_table)
        dropQuery.execute_query()

        createQuery = self.qb.create_createTable_query(new_table).like(old_table)
        createQuery.execute_query()
        
        self.data_engine.disable_table_keys(new_table)
        
        selectQuery = self.qb.create_select_query(old_table).set_fields(['*']).where(where) 
        insertQuery = self.qb.create_insert_query(new_table).values_from_select(selectQuery)
        insertQuery.execute_query()
        
        self.data_engine.enable_table_keys(new_table)

        return new_table
    
    def makeBlackWhiteList(self, args_featlist, args_lextable, args_categories):
        """?????
 
        Parameters
        ----------
        args_featlist : str
            ?????
        args_lextable : str
            ?????
        args_categories : str
            ?????
     
        Returns
        -------
        newlist : list
            ?????
        """
        newlist = set()
        if self.use_unicode:
            print("making black or white list: [%s] [%s] [%s]" %([feat if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories))
        else:
            print("making black or white list: [%s] [%s] [%s]" %([feat if isinstance(feat, str) else feat for feat in args_featlist], args_lextable, args_categories))
        if args_lextable:
            
            fields = ["term"]
            if (len(args_categories) > 0) and args_categories[0] != '*':
                where_condition = "category in (%s)" % (args_lextable, ','.join(['\''+str(x)+'\'' for x in args_categories]))
            else:
                where_condition = ''
           
            if self.lexicon is None: self.load_lexicon(args_lextable) 
            selectQuery = self.lexicon.qb.create_select_query(args_lextable).where(where_condition).set_fields(fields)
            rows = selectQuery.execute_query()

            for row in rows:
                newlist.add(row[0])
        elif args_featlist:
            for feat in args_featlist:
                if self.use_unicode:
                    feat = feat if isinstance(feat, str) else feat
                else:
                    feat = feat if isinstance(feat, str) else feat
                # newlist.add(feat.lower())
                if self.use_unicode:
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

