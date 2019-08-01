import sys
import re
import MySQLdb
import pandas as pd
import numpy as np
from math import isclose
from configparser import SafeConfigParser

from .dlaWorker import DLAWorker
from . import dlaConstants as dlac
from .mysqlmethods import mysqlMethods as mm
from .mysqlmethods.mysql_iter_funcs import get_db_engine

class OutcomeGetter(DLAWorker):
    """Deals with outcome tables

    Parameters
    ----------
    outcome_table : str
        MySQL table name

    outcome_value_fields : list
        List of MySQL column names to be used as outcomes 

    outcome_controls : list
        List of MySQL column names to be used as controls 

    outcome_interaction : list
        List of MySQL column names to be used as interactions 

    group_freq_thresh : str
        Minimum word threshold per group

    featureMapping : str
        ?????

    one_group_set_for_all_outcomes : str
        ?????

    fold_column : str
        ?????

    low_variance_thresh : float
        Threshold to remove low variance outcomes or controls from analysis

    Returns
    -------
    OutcomeGetter object

    Examples
    --------
    Initialize a OutcomeGetter

    >>> og = OutcomeGetter.fromFile('~/myInit.ini')
    
    Get outcome table as pandas dataframe

    >>> outAndCont = og.getGroupsAndOutcomesAsDF()


    """

    _mysqlNumeric = set(['TINYINT', 'SMALLINT', 'MEDIUMINT','INT', 'INTEGER', 'BIGINT','FLOAT',
                         'DOUBLE', 'DOUBLE PRECISION','REAL','DECIMAL','NUMERIC'])
    _mysqlDate = set(['DATE','DATETIME','TIMESTAMP','TIME','YEAR'])

    @classmethod
    def fromFile(cls, initFile):
        """Loads specified features from file

        Parameters
        ----------
        initFile : str 
            Path to file

        """
        parser = SafeConfigParser()
        parser.read(initFile)
        corpdb = parser.get('constants','corpdb') if parser.has_option('constants','corpdb') else dlac.DEF_CORPDB
        corptable = parser.get('constants','corptable') if parser.has_option('constants','corptable') else dlac.DEF_CORPTABLE
        correl_field = parser.get('constants','correl_field') if parser.has_option('constants','correl_field') else dlac.DEF_CORREL_FIELD
        mysql_host = parser.get('constants','mysql_host') if parser.has_option('constants','mysql_host') else dlac.MYSQL_HOST
        message_field = parser.get('constants','message_field') if parser.has_option('constants','message_field') else dlac.DEF_MESSAGE_FIELD
        messageid_field = parser.get('constants','messageid_field') if parser.has_option('constants','messageid_field') else dlac.DEF_MESSAGEID_FIELD
        encoding = parser.get('constants','encoding') if parser.has_option('constants','encoding') else dlac.DEF_ENCODING
        if parser.has_option('constants','use_unicode'):
            use_unicode = True if parser.get('constants','use_unicode')=="True" else False
        else:
            use_unicode = dlac.DEF_UNICODE_SWITCH
        lexicondb = parser.get('constants','lexicondb') if parser.has_option('constants','lexicondb') else dlac.DEF_LEXICON_DB
        outcome_table = parser.get('constants','outcometable') if parser.has_option('constants','outcometable') else dlac.DEF_OUTCOME_TABLE
        outcome_value_fields = [o.strip() for o in parser.get('constants','outcomefields').split(",")] if parser.has_option('constants','outcomefields') else [dlac.DEF_OUTCOME_FIELD] # possible list
        outcome_controls = [o.strip() for o in parser.get('constants','outcomecontrols').split(",")] if parser.has_option('constants','outcomecontrols') else dlac.DEF_OUTCOME_CONTROLS # possible list
        outcome_interaction = [o.strip() for o in parser.get('constants','outcomeinteraction').split(",")] if parser.has_option('constants','outcomeinteraction') else dlac.DEF_OUTCOME_CONTROLS # possible list
        outcome_categories = [o.strip() for o in parser.get('constants','outcomecategories').split(",")] if parser.has_option('constants','outcomecategories') else [] # possible list
        multiclass_outcome = [o.strip() for o in parser.get('constants','multiclassoutcome').split(",")] if parser.has_option('constants','multiclassoutcome') else [] # possible list
        group_freq_thresh = parser.get('constants','groupfreqthresh') if parser.has_option('constants','groupfreqthresh') else dlac.getGroupFreqThresh(correl_field)
        low_variance_thresh = parser.get('constants','lowvariancethresh') if parser.has_option('constants','lowvariancethresh') else dlac.DEF_LOW_VARIANCE_THRESHOLD
        featureMappingTable = parser.get('constants','featlabelmaptable') if parser.has_option('constants','featlabelmaptable') else ''
        featureMappingLex = parser.get('constants','featlabelmaplex') if parser.has_option('constants','featlabelmaplex') else ''
        wordTable = parser.get('constants','wordTable') if parser.has_option('constants','wordTable') else None
        return cls(corpdb=corpdb, corptable=corptable, correl_field=correl_field, mysql_host=mysql_host, message_field=message_field, messageid_field=messageid_field, encoding=encoding, use_unicode=use_unicode, lexicondb=lexicondb, outcome_table=outcome_table, outcome_value_fields=outcome_value_fields, outcome_controls=outcome_controls, outcome_interaction=outcome_interaction, outcome_categories=outcome_categories, multiclass_outcome=multiclass_outcome, group_freq_thresh=group_freq_thresh, low_variance_thresh=low_variance_thresh, featureMappingTable=featureMappingTable, featureMappingLex=featureMappingLex, wordTable=wordTable)
    

    def __init__(self, corpdb=dlac.DEF_CORPDB, corptable=dlac.DEF_CORPTABLE, correl_field=dlac.DEF_CORREL_FIELD, mysql_host=dlac.MYSQL_HOST, message_field=dlac.DEF_MESSAGE_FIELD, messageid_field=dlac.DEF_MESSAGEID_FIELD, encoding=dlac.DEF_ENCODING, use_unicode=dlac.DEF_UNICODE_SWITCH, lexicondb = dlac.DEF_LEXICON_DB, outcome_table=dlac.DEF_OUTCOME_TABLE, outcome_value_fields=[dlac.DEF_OUTCOME_FIELD], outcome_controls = dlac.DEF_OUTCOME_CONTROLS, outcome_interaction = dlac.DEF_OUTCOME_CONTROLS, outcome_categories = [], multiclass_outcome = [], group_freq_thresh = None, low_variance_thresh = dlac.DEF_LOW_VARIANCE_THRESHOLD, featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None):
        super(OutcomeGetter, self).__init__(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, wordTable = wordTable)
        self.outcome_table = outcome_table

        if isinstance(outcome_value_fields, str):
            outcome_value_fields = [outcome_value_fields]

        if outcome_value_fields and len(outcome_value_fields) > 0 and outcome_value_fields[0] == '*':#handle wildcard fields
            newOutcomeFields = []
            for name, typ in mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, outcome_table).items():
                typ = re.sub(r'\([0-9\,]*\)\s*$', '', typ)
                if typ.split()[0].upper() in self._mysqlNumeric:
                    newOutcomeFields.append(name)
            outcome_value_fields = newOutcomeFields

        if outcome_controls and len(outcome_controls) > 0 and outcome_controls[0] == '*':#handle wildcard fields
            newOutcomeFields = []
            for name, typ in mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, outcome_table).items():
                typ = re.sub(r'\([0-9\,]*\)\s*$', '', typ)
                if typ.split()[0].upper() in self._mysqlNumeric:
                    newOutcomeFields.append(name)
            outcome_controls = newOutcomeFields
        
        self.outcome_value_fields = outcome_value_fields
        self.outcome_controls = outcome_controls
        self.outcome_interaction = outcome_interaction
        self.outcome_categories = outcome_categories
        self.multiclass_outcome = multiclass_outcome
        #if not group_freq_thresh and group_freq_thresh != 0:
        if group_freq_thresh is None:
            self.group_freq_thresh = dlac.getGroupFreqThresh(self.correl_field)
        else:
            self.group_freq_thresh = group_freq_thresh
        self.featureMapping = self.getFeatureMapping(featureMappingTable, featureMappingLex, False)
        self.one_group_set_for_all_outcomes = False # whether to use groups in common for all outcomes
        self.fold_column = fold_column
        self.low_variance_thresh = low_variance_thresh

    def hasOutcomes(self):
        if len(self.outcome_value_fields) > 0:
            return True
        return False

    def copy(self):
        self.__dict__
        newObj = OutcomeGetter(self.corpdb, self.corptable, self.correl_field, self.mysql_host, self.message_field, self.messageid_field)
        for k, v in self.__dict__.items():
            newObj.__dict__[k] = v
        return newObj

    def getFeatureMapping(self, featureMappingTable, featureMappingLex, bracketlabels):
        feat_to_label = {}
        if featureMappingTable:
            feat_to_label = self.getLabelmapFromLabelmapTable(featureMappingTable)
        elif featureMappingLex:
            feat_to_label = self.getLabelmapFromLexicon(featureMappingLex)
            assert( featureMappingTable != featureMappingLex )

        if bracketlabels:
            for feat, label in feat_to_label.items():
                feat_to_label[feat] = '{' + label + '}'
            
        return feat_to_label

    def createOutcomeTable(self, tablename, dataframe, ifExists='fail'):
        eng = get_db_engine(self.corpdb, self.mysql_host)
        dtype ={}
        if isinstance(dataframe.index[0], str):
            import sqlalchemy
            dataframe.index = dataframe.index.astype(str)
            dtype = {self.correl_field : sqlalchemy.types.VARCHAR(max([len(i) for i in dataframe.index]))}
            dataframe.index.name = self.correl_field
        dataframe.to_sql(tablename, eng, index_label = self.correl_field, if_exists = ifExists, chunksize=dlac.MYSQL_BATCH_INSERT_SIZE, dtype = dtype)
        print("New table created: %s" % tablename)

    def getDistinctOutcomeValues(self, outcome = None, includeNull = True, where = ''):
        """returns a list of outcome values"""
        if not outcome:
            outcome = self.outcome_value_fields[0]
        sql = "SELECT DISTINCT %s FROM %s"%(outcome, self.outcome_table)
        if not includeNull or where: 
            wheres = []
            if where: wheres.append(where)
            if not includeNull:
                wheres.append("%s IS NOT NULL" % outcome)
            sql += ' WHERE ' + ' AND '.join(wheres)
        return [v[0] for v in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding)]

    def getDistinctOutcomeValueCounts(self, outcome = None, requireControls = False, includeNull = True, where = ''):
        """returns a dict of (outcome_value, count)"""
        if not outcome:
            outcome = self.outcome_value_fields[0]
        sql = "SELECT %s, count(*) FROM %s"%(outcome, self.outcome_table)
        if requireControls or not includeNull or where: 
            wheres = []
            if where: wheres.append(where)
            if requireControls:
                for control in self.outcome_controls:
                    wheres.append("%s IS NOT NULL" % control)
            if not includeNull:
                wheres.append("%s IS NOT NULL" % outcome)
            sql += ' WHERE ' + ' AND '.join(wheres)
            
        sql += ' group by %s ' % outcome
        return dict(mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode))

    def getDistinctOutcomeAndControlValueCounts(self, outcome = None, control = None, includeNull = True, where = ''):
        """returns a dict of (outcome_value, count)"""
        #TODO: muliple controls
        if not outcome:
            outcome = self.outcome_value_fields[0]
        if not control:
            control = self.outcome_controls[0]

        sql = "SELECT %s, %s, count(*) FROM %s"%(outcome, control, self.outcome_table)
        if not includeNull or where: 
            wheres = []
            if where: wheres.append(where)
            if not includeNull:
                wheres.append("%s IS NOT NULL" % outcome)
                wheres.append("%s IS NOT NULL" % control)
            sql += ' WHERE ' + ' AND '.join(wheres)
            
        sql += ' group by %s, %s ' % (outcome, control)
        countDict = dict()
        for (outcome, control, count) in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode):
            if not outcome in countDict:
                countDict[outcome] = dict()
            countDict[outcome][control] = count
        return countDict

        
    def getGroupAndOutcomeValues(self, outcomeField = None, where=''):
        """returns a list of (group_id, outcome_value) tuples"""
        if not outcomeField: outcomeField = self.outcome_value_fields[0]
        sql = "select %s, %s from `%s` WHERE %s IS NOT NULL"%(self.correl_field, outcomeField, self.outcome_table, outcomeField)
        if (where): sql += ' AND ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

    def makeContingencyTable(self, featureGetter, featureValueField, outcome_filter_where='', feature_value_group_sum_min=0):
        """makes a contingency table from this outcome value, a featureGetter, and the desired column of the featureGetter, assumes both correl_field's are the same"""
        """follows http://www.artfulsoftware.com/infotree/queries.php at section: Group Column Statistics in Rows"""
        """the only time this uses outcome_value's would be in the outcome_filter_where statement"""
        fg = featureGetter
        distinctFeatureList = fg.getDistinctFeatures() #access single idx
        featZeroDict = dict(fg.getFeatureZeros())
        
        sql = "SELECT %s, "%(fg.correl_field)

        def makeCaseStrings( distinctFeature ):
            df = distinctFeature[0]
            zero = .0000001
            if df in featZeroDict:
                zero = featZeroDict[df]
            df = MySQLdb.escape_string(df)
            if df:#debug
                return "( CASE feat WHEN '%s' THEN %s ELSE %s END ) AS '%s'"%(df, featureValueField, str(zero), df)
            return ''

        case_statements = list(map(makeCaseStrings, distinctFeatureList))
        sql_cases_features = ", ".join(case_statements) + " "
        #debugN = 1000 #DEBUG
        #_warn( distinctFeatureList[0:debugN] ) #DEBUG
        #sql_cases_features = "".join(case_statements[0:debugN]) #DEBUG
        
        # update the main sql statement to select distinct features as columns
        sql += sql_cases_features

        # filter out the outcomes based on the outcome_filter_where statement
        # an example would be outcome_filter_where = "self.featureValueField is not null and self.featureValueField > 0.50"
        sql_new_outcome_correl_ids = "( SELECT %s FROM %s "%(self.correl_field, self.outcome_table)
        if outcome_filter_where: sql_new_outcome_correl_ids += "WHERE " + outcome_filter_where
        sql_new_outcome_correl_ids += ")"

        # This piece takes care of "users with > 4000 words"
        sql_filtered_feature_table = fg.featureTable
        if feature_value_group_sum_min > 0:
            # Get a sum of "value" for each group_id
            sql_group_ids_and_value_counts = "( SELECT %s, SUM(value) AS value_sum FROM %s GROUP BY %s )"%(fg.correl_field, fg.featureTable, fg.correl_field)
            # Select group_id that have a "value_sum" >= N (input as a function argument; feature_value_group_sum_min)
            sql_group_ids_with_sufficient_value = "( SELECT %s FROM %s AS groupIdsAndSums WHERE value_sum > %s )"%(fg.correl_field, sql_group_ids_and_value_counts, feature_value_group_sum_min)
            # Select the subset of the original fg.featureTable where group_id meets the "value_sum >= N" condition
            sql_filtered_feature_table = "( SELECT featuresOriginal.* FROM %s AS featuresOriginal, %s AS featuresSubset WHERE featuresOriginal.%s = featuresSubset.%s )"%( fg.featureTable, sql_group_ids_with_sufficient_value, fg.correl_field, fg.correl_field)


        # update the feature table to contain only the outcomes from the filtered id's
        sql_filtered_feature_table_2 = "( SELECT filteredFeatures.* FROM %s AS filteredFeatures, %s AS filteredOutcomes WHERE filteredFeatures.%s = filteredOutcomes.%s)"%(sql_filtered_feature_table, sql_new_outcome_correl_ids, fg.correl_field, self.correl_field)

        # finish the original query with the updated feature table
        sql += "FROM %s AS updatedFeatures GROUP BY %s"%(sql_filtered_feature_table_2, fg.correl_field)
        return [distinctFeatureList, mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)]

    def makeBinnedOutcomeTable(self, buckets, mid_aom_list):
        """buckets is a list of tuples"""
        raise NotImplementedError

    def getGroupsAndOutcomes(self, lexicon_count_table=None, groupsWhere = '', includeFoldLabels=False):
        if self.group_freq_thresh and self.wordTable != self.get1gramTable():
            dlac.warn("""You specified a --word_table and --group_freq_thresh is
enabled, so the total word count for your groups might be off
(remove "--word_table WT" to solve this issue)""", attention=False)
        
        if self.outcome_table:
            self.checkIndices(self.outcome_table, primary=self.correl_field, correlField=self.correl_field)

        groups = set()
        outcomes = dict()
        outcomeFieldList = set(self.outcome_value_fields).union(set(self.outcome_controls)).union(set(self.outcome_interaction))
        ocs = dict()
        controls = dict()

        #get outcome values:
        dlac.warn("Loading Outcomes and Getting Groups for: %s" % str(outcomeFieldList)) #debug
        if outcomeFieldList:
            to_remove = []
            for outcomeField in outcomeFieldList:
                outcomes[outcomeField] = dict(self.getGroupAndOutcomeValues(outcomeField, where=groupsWhere))
                if self.low_variance_thresh is not None and self.low_variance_thresh is not False:
                    try:
                        outcomeVariance = np.var(list(outcomes[outcomeField].values()))
                        if isclose(outcomeVariance, 0.0) or outcomeVariance < self.low_variance_thresh:
                            del outcomes[outcomeField]
                            dlac.warn("Removing %s from analysis: variance %s less than threshold %s. To keep use --keep_low_variance" % (outcomeField, outcomeVariance, self.low_variance_thresh))
                            to_remove.append(outcomeField)
                            continue
                    except TypeError:
                        dlac.warn("TypeError during variance check for %s, skipping step." % (outcomeField))
                        outcomeVariance = 1
                    if isclose(outcomeVariance, 0.0) or outcomeVariance < self.low_variance_thresh:
                        del outcomes[outcomeField]
                        dlac.warn("Removing %s from analysis: variance %s less than threshold %s. To keep use --keep_low_variance" % (outcomeField, outcomeVariance, self.low_variance_thresh))
                        to_remove.append(outcomeField)
                        continue


                if outcomeField in self.outcome_value_fields:
                    groups.update(list(outcomes[outcomeField].keys()))
            for outcome in to_remove: 
                outcomeFieldList.remove(outcome)
                if outcome in self.outcome_value_fields: self.outcome_value_fields.remove(outcome)
                elif outcome in self.outcome_controls: self.outcome_controls.remove(outcome)
                elif outcome in self.outcome_interaction: self.outcome_interaction.remove(outcome)
            if len(self.outcome_value_fields) == 0: 
                dlac.warn("No outcomes remaining after checking variances.")
                sys.exit(1)

            # create one hot representation of outcome
            if self.outcome_categories:
                for cat in self.outcome_categories:
                    cat_label_list = []
                    try:
                        if not all(isinstance(lbl, (int, str)) for lbl in outcomes[cat].values()):
                            dlac.warn("Arguments of --categories_to_binary must contain string or integer values")
                            sys.exit(1)
                        cat_labels = set([str(lbl) for lbl in outcomes[cat].values()])
                        if len(cat_labels) == 2: cat_labels.pop()
                        for lbl in cat_labels:
                            cat_label_str = "__".join([cat, lbl]).replace(" ", "_").lower()
                            outcomes[cat_label_str] = {gid:1 if str(l) == lbl else 0 for gid, l in outcomes[cat].items() }
                            cat_label_list.append(cat_label_str)
                    except:
                        dlac.warn("Arguments of --categories_to_binary do not match --outcomes or --controls")
                        sys.exit(1)
                    del outcomes[cat]
                    if cat in self.outcome_value_fields:
                        self.outcome_value_fields.remove(cat)
                        self.outcome_value_fields += cat_label_list
                    elif cat in self.outcome_controls:
                        self.outcome_controls.remove(cat)
                        self.outcome_controls += cat_label_list
                    else:
                        self.outcome_interaction.remove(cat)
                        self.outcome_interaction += cat_label_list
            
            # create multiclass (integer) representation of outcome
            if self.multiclass_outcome:
                cat_labels_dict = dict() # store the final mapping in self.multiclass_outcome
                for moutcome in self.multiclass_outcome:
                    cat_label_list = []
                    try:
                        if not all(isinstance(lbl, (str)) for lbl in outcomes[moutcome].values()):
                            dlac.warn("Arguments of --multiclass must contain only string values")
                            sys.exit(1)
                        cat_labels = set([str(lbl).lower() for lbl in outcomes[moutcome].values()])
                        cat_label_str = "_".join([moutcome, "_multiclass"]).lower()
                        cat_labels_dict[cat_label_str] = {i[1]:i[0] for i in enumerate(sorted(cat_labels))}
                        outcomes[cat_label_str] = {gid: cat_labels_dict[cat_label_str][l.lower()] for gid, l in outcomes[moutcome].items() }
                        cat_label_list.append(cat_label_str)
                    except:
                        dlac.warn("Arguments of --multiclass do not match --outcomes or --controls")
                        sys.exit(1)
                    del outcomes[moutcome]
                    if moutcome in self.outcome_value_fields:
                        self.outcome_value_fields.remove(moutcome)
                        self.outcome_value_fields += cat_label_list
                    elif moutcome in self.outcome_controls:
                        self.outcome_controls.remove(moutcome)
                        self.outcome_controls += cat_label_list
                    else:
                        self.outcome_interaction.remove(moutcome)
                        self.outcome_interaction += cat_label_list
                self.multiclass_outcome = cat_labels_dict

            if self.group_freq_thresh:
                where = """ group_id in ('%s')""" % ("','".join(str(g) for g in groups))
                groupCnts = self.getGroupWordCounts(where, lexicon_count_table = lexicon_count_table)
                groups = set()
                for outcomeField, outcomeValues in outcomes.items():
                    newOutcomes = dict()
                    for gId in outcomeValues.keys():
                        if (gId in groupCnts) and (groupCnts[gId] >= self.group_freq_thresh):
                            #keep
                            # newOutcomes[gId] = float(outcomeValues[gId])
                            newOutcomes[gId] = outcomeValues[gId]
                    outcomes[outcomeField] = newOutcomes
                    if outcomeField in self.outcome_value_fields:
                        groups.update(list(newOutcomes.keys()))

            #set groups:
            for k in self.outcome_controls + self.outcome_interaction:
                groups = groups & set(outcomes[k].keys()) #always intersect with controls
            if self.outcome_categories:
                for cat in cat_label_list:
                    if all(outcomes[cat][group] == 0 for group in groups):
                        del outcomes[cat]
                        dlac.warn("Removing %s, no non-zero instances" % cat)
                        if cat in self.outcome_value_fields:
                            self.outcome_value_fields.remove(cat)
                        elif cat in self.outcome_controls:
                            self.outcome_controls.remove(cat)
                        else:
                            self.outcome_interaction.remove(cat)

            if groupsWhere:
                outcm = groupsWhere.split()[0].strip()
                whereusers = set([i[0] for i in self.getGroupAndOutcomeValues(outcm, where=groupsWhere)])
                groups = groups & whereusers

            if self.one_group_set_for_all_outcomes:
                for k in self.outcome_value_fields:
                    groups = groups & set(outcomes[k].keys()) # only intersect if wanting all the same groups
            
            #split into outcomes and controls:
            ocs = dict()
            controls = dict()
            for k in self.outcome_controls + self.outcome_interaction:
                outcomeDict = outcomes[k]
                outcomeDict = dict([(g, v) for g, v in outcomeDict.items() if g in groups])
                controls[k] = outcomeDict
            for k in self.outcome_value_fields:
                outcomeDict = outcomes[k]
                outcomeDict = dict([(g, v) for g, v in outcomeDict.items() if g in groups])
                ocs[k] = outcomeDict
        elif self.group_freq_thresh:
            groupCnts = self.getGroupWordCounts(where = None, lexicon_count_table = lexicon_count_table)
            groups = set()
            for gId, cnt in groupCnts.items():
                if cnt >= self.group_freq_thresh:
                    groups.add(gId)
            if groupsWhere:
                outcm = groupsWhere.split('=')[0].strip()
                val = groupsWhere.split('=')[1].strip()
                whereusers = set([i[0] for i in self.getGroupAndOutcomeValues(outcm) if str(i[1]) == val])
                groups = groups & whereusers

        if self.fold_column:
            folds = dict(self.getGroupAndOutcomeValues(self.fold_column))
        else:
            folds = None

        if includeFoldLabels:
            return (groups, ocs, controls, folds)
        else:
            return (groups, ocs, controls)

    def getGroupAndOutcomeValuesAsDF(self, outcomeField = None, where=''):
        """returns a dataframe of (group_id, outcome_value)"""
        if not outcomeField: outcomeField = self.outcome_value_fields[0]
        db_eng = get_db_engine(self.corpdb)
        sql = "select %s, %s from `%s` WHERE %s IS NOT NULL" % (self.correl_field, outcomeField, self.outcome_table, outcomeField)
        if (where): sql += ' WHERE ' + where
        index = self.correl_field
        return pd.read_sql(sql=sql, con=db_eng, index_col=index)

    def getGroupsAndOutcomesAsDF(self, lexicon_count_table=None, groupsWhere = '', sparse=False):
        (groups, allOutcomes, controls) = self.getGroupsAndOutcomes(lexicon_count_table, groupsWhere)
        o_df = pd.DataFrame(allOutcomes)
        c_df = pd.DataFrame(controls)
        if sparse:
            df = pd.concat([o_df, c_df], axis=1).to_sparse(fill_value=0)
            df.index.names = ['group_id']
            return df
        else:
            df = pd.concat([o_df, c_df], axis=1)
            df.index.names = ['group_id']
            return df

    def numGroupsPerOutcome(self, featGetter, outputfile, where = ''):
        """prints sas-style csv file output"""
        #get outcome data to work with
        (groups, allOutcomes, controls) = self.getGroupsAndOutcomes()

        #adjust keys for outcomes and controls:
        countGroups = dict()
        for outcomeField, outcomes in allOutcomes.items():
            countGroups[outcomeField] = len(outcomes)

        return countGroups
