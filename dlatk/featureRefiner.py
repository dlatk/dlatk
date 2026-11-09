import re
from collections import OrderedDict
from pprint import pprint
import pandas as pd
import sys
from dateutil.parser import parse as dtParse
import datetime
import time

#math / stats:
from numpy import sqrt, array, std, mean, log2, log
from numpy.linalg import norm
import numpy as np
import math
import numbers
from operator import mul
from functools import reduce
from sklearn.preprocessing import StandardScaler
try:
    from scipy.sparse import csr_array
except ImportError as e:
    from scipy.sparse import csr_matrix as csr_array
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import interp1d

#local / nlp
from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer

from .featureGetter import FeatureGetter
from . import smoothing
from . import dlaConstants as dlac
from .mysqlmethods import mysqlMethods as mm
from .mysqlmethods import mysql_iter_funcs as mif

from .database.query import QueryBuilder
from .database.query import Column
from .database.dataEngine import DataEngine

from .regressionPredictor import alignDictsAsXy


def splitHalfFromSums(sx, sy, sxx, syy, sxy, n):
    """Spearman-Brown corrected correlation, 2r / (1 + r), between two halves x and y over n groups, from the sums
    of x, y, x^2, y^2 and xy (arrays); 0 where a half does not vary. Negative r gives a negative reliability."""
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = sxy / n - (sx / n) * (sy / n)
        r = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    r = np.clip(np.nan_to_num(r), -0.999, 1.0)
    return 2 * r / (1 + r)


def nbReliability(sumC, sumC2, sumCN, sumN, sumN2, meanInvN):
    """reliability across groups of features' relative frequencies under a gamma-Poisson (negative binomial) model,
    assuming words are independent draws: true rates vary across groups with variance p^2/a, and a group with N words
    adds sampling variance p/N, so reliability = (p^2/a) / (p^2/a + p * mean_g(1/N_g)) -- the same quantity the
    split-half estimates (how much the feature's correlations are attenuated, by sqrt(reliability)). Per feature
    (arrays): sums over groups of c, c^2 and c*N (c = the feature's count); for the feature's n-gram size: sums over
    groups of N and N^2, and the mean of 1/N over groups."""
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.nan_to_num(sumC / sumN)
        excess = sumC2 - 2 * p * sumCN + p * p * sumN2 - p * sumN #sum over groups of (c - N*p)^2 - N*p
        a = np.where(excess > 0, np.clip(p * p * sumN2 / excess, smoothing.A_MIN, smoothing.A_MAX), smoothing.A_MAX)
        between = p * p / a
        rel = np.nan_to_num(between / (between + p * meanInvN))
    return np.where(p > 0, rel, 0.0)


def groupReliability(reliability, n, harmonicN):
    """reliability of one group's relative frequency of a feature, from the feature's reliability across groups:
    sampling noise is taken to shrink as 1/N (N = the group's total n-grams), so a group with N n-grams gets the
    Spearman-Brown projection of the across-group reliability (which is that of a group with the harmonic mean of
    the groups' totals) to length N / harmonicN:  rel_g = rel / (rel + (1 - rel) * harmonicN / N). Under the negative
    binomial model this is exactly (p^2/a) / (p^2/a + p/N). Negative reliabilities are taken as 0; 0 when N is 0.
    Noise shrinking as 1/N is an approximation when groups' messages differ in length."""
    rel = np.clip(reliability, 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = rel / (rel + (1.0 - rel) * harmonicN / n)
    return np.where((n > 0) & (rel > 0), np.nan_to_num(out), 0.0)


class FeatureRefiner(FeatureGetter):
    """Deals with the refinement of feature information already in a table (outputs to new table)

    Returns
    -------
    FeatureRefiner object
    """

    def makeTopicLabelMap(self, topiclexicon, numtopicwords=5, is_weighted_lexicon=False):
        featlabel_tablename = 'feat_to_label$%s$%d'%(topiclexicon, numtopicwords)

        pldb = self.lexicondb
        (plconn, plcur, plcurD) = mm.dbConnect(pldb, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = 'DROP TABLE IF EXISTS `%s`'%featlabel_tablename
        mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = 'CREATE TABLE `%s` (`id` int(16) unsigned NOT NULL AUTO_INCREMENT, `term` varchar(128) DEFAULT NULL, `category` varchar(64) DEFAULT NULL, PRIMARY KEY (`id`), KEY `term` (`term`), KEY `category` (`category`) ) CHARACTER SET %s COLLATE %s ENGINE=%s' % (featlabel_tablename, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()], dlac.DEF_MYSQL_ENGINE)
        mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        sql = 'SELECT DISTINCT category FROM %s'%topiclexicon
        categories = [x[0] for x in mm.executeGetList(pldb, plcur, sql)]
        label_list = []
        for category in categories:
            if is_weighted_lexicon:
                sql = 'SELECT term, weight from %s WHERE category = \'%s\''%(topiclexicon, category)
                rows = mm.executeGetList(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                top_n_rows = sorted(rows, key=lambda x:x[1], reverse=True)
                terms = [x[0] for x in top_n_rows]
                label = ' '.join(map(str, terms[0:numtopicwords]))
                escaped_label = MySQLdb.escape_string(label)
                sql = 'INSERT INTO `%s` (`term`, `category`) VALUES(\'%s\', \'%s\')'%(featlabel_tablename, category, escaped_label )
                mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            else:
                sql = 'SELECT term from %s WHERE category = \'%s\''%(topiclexicon, category)
                terms = [x[0] for x in mm.executeGetList(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode)]
                label = ' '.join(map(str, terms[0:numtopicwords]))
                escaped_label = MySQLdb.escape_string(label)
                sql = 'INSERT INTO `%s` (`term`, `category`) VALUES(\'%s\', \'%s\')'%(featlabel_tablename, category, escaped_label )
                mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return featlabel_tablename

    def createCombinedFeatureTable(self, featureName = None, featureTables = [], tableName = None, appendWithNewID = False):
        """Create a new feature table by combining others
        appendWithNewID: will update the group id for each table so appending rather than union
        """
        

        #get best feat column type:
        intGrabber = re.compile(r'\d+')
        longestInt = 12
        longestType = "VARCHAR(12)"
        valueType = 'INTEGER'
        for table in featureTables:
            columns = self.data_engine.getTableColumnNameTypes(table)
            if not columns: raise ValueError("One of your feature tables probably doesn't exist")
            currentType = columns['feat']
            valueType = columns['value']
            currentInt = int(intGrabber.search(currentType).group())
            if currentInt > longestInt:
                longestInt = currentInt
                longestType = currentType

        #get transformation:
        toNum = None
        numMatch = re.search(r'16to(\d+)', featureTables[0])
        if numMatch:
            toNum = numMatch.group(1)
        valueFunc = None
        if toNum:
            for func in dlac.POSSIBLE_VALUE_FUNCS:
                if round(float(func(16))) == float(toNum):
                    valueFunc = func
                    break     
        pocc = None
        poccmatch = re.search(r'\$([_0-9]+)\s*($|\$)', featureTables[0])
        if poccmatch:
            pocc = poccmatch.group(1)
        
        #CREATE TABLE:
        if not featureName:
            names = []
            featNameGrabber = re.compile(r'^feat\$([^\$]+)\$')
            for table in featureTables:
                names.append(featNameGrabber.match(table).group(1))
            featureName = '_'.join(names)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%longestInt, valueType, tableName, valueFunc, extension=pocc, correl_fieldType = 'VARCHAR(120)' if appendWithNewID else None)
        # Maarten: todo: test if too long and don't disable keys
        self.data_engine.disable_table_keys(featureTableName) #for faster, when enough space for repair by sorting
        
        insert_idx_start = 0
        insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE
        tnum = 0 #for appending
        for fTable in featureTables:
            if self.db_type == "sqlite":
                insertQuery = self.qb.create_insert_query(featureTableName).set_values([("group_id", ""),("feat", ""),("value", ""),("group_norm", "")])
                selectQuery = self.qb.create_select_query(fTable).set_fields(["group_id", "feat", "value", "group_norm"])
                rows = selectQuery.execute_query()
                if appendWithNewID:
                    rows = [[str(tnum)+"_"+str(r[0])]+r[1:] for r in rows]
                insertQuery.execute_query(rows)
            elif self.db_type == "mysql":
                sql = "INSERT INTO %s (group_id, feat, value, group_norm) SELECT group_id, feat, value, group_norm from %s;"%(featureTableName, fTable)
                if appendWithNewID:#appends tables instead of union
                    sql = "INSERT INTO %s (group_id, feat, value, group_norm) SELECT CONCAT('%s_',group_id), feat, value, group_norm from %s;"%(featureTableName, tnum, fTable)
                self.data_engine.execute(sql)
            tnum += 1
        
        self.data_engine.enable_table_keys(featureTableName)

        return featureTableName


    def createTableWithBinnedFeats(self, num_bins, group_id_range, valueFunc = lambda x:x, 
                                   gender=None, genderattack=False, reporting_percent=0.04, outcomeTable = dlac.DEF_OUTCOME_TABLE, skip_binning=False):
        featureTable = self.featureTable
        group_id_range = list(map(int, group_id_range))
        newTable = featureTable+'$'+str(num_bins)+'b_'+'_'.join(map(str,group_id_range))
        if skip_binning: return newTable

        sql = 'DROP TABLE IF EXISTS %s'%newTable
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = "CREATE TABLE %s like %s" % (newTable, featureTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.standardizeTable(self.corpdb, self.dbCursor, newTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        groupNs = mm.executeGetList(self.corpdb, self.dbCursor, 'SELECT group_id, N FROM %s GROUP BY group_id'%self.featureTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        groupIdToN = dict(groupNs)
        total_freq = sum([x[1] for x in groupNs])
        bin_size = float(total_freq) / float(num_bins+2)

        num_groups = len(groupNs)
        reporting_int = dlac._getReportingInt(reporting_percent, num_groups)

        # figure out the bins, i.e. if group_id's 1,2,3 total value is greater than "bin_size" our first bin is 1_3.
        dlac.warn('determining the number of bins...')
        current_sum = 0
        current_lower_group = groupNs[0][0]

        current_upper_group = None
        next_group_is_lower_group = False
        bin_groups = OrderedDict()
        gg = 0
        for group, value in groupNs:
            if next_group_is_lower_group:
                current_lower_group = group
                next_group_is_lower_group = False
            current_sum += value
            current_upper_group = group
            if current_sum >= bin_size:
                current_sum = 0
                bin_groups[(current_lower_group, current_upper_group)]  = '_'.join(map(str,[current_lower_group, current_upper_group]))
                next_group_is_lower_group = True
            gg += 1
            dlac._report('group_id\'s', gg, reporting_int, num_groups)
        if current_sum >= 0:
            bin_groups[(current_lower_group, current_upper_group)]  = '_'.join(map(str,[current_lower_group, current_upper_group]))

        max_label_length = max(list(map(len, list(bin_groups.values()))))

        sql = 'ALTER TABLE %s MODIFY COLUMN group_id VARCHAR(%d)'%(newTable, max_label_length) #this action preserves the index
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = 'ALTER TABLE %s ADD COLUMN `bin_center` float(6) not null default -1.0'%(newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = 'ALTER TABLE %s ADD COLUMN `bin_center_w` float(6) not null default -1.0'%(newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = 'ALTER TABLE %s ADD COLUMN `bin_width` int(10) not null default -1'%(newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.disableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        # for each newly denoted bin: e.g. 1_3, 4_5, 6_6, ... get the new feature value counts / group norms; insert them into the new table
        # e.g. 1 'hi' 5, 2 'hi' 10, 3 'hi' 30 ==> 1_3 'hi' 45  (of course include group_norm also)
        dlac.warn('aggreagating the newly binned feature values / group_norms into the new table...')
        isql = 'INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N, bin_center, bin_center_w, bin_width) VALUES (%s)'%(newTable, '%s, %s, %s, %s, %s, %s, %s, %s, %s')
        #isql = 'INSERT INTO %s (group_id, feat, value, group_norm, N, bin_center, bin_width) VALUES (%s)'%(newTable, '%s, %s, %s, %s, %s, %s, %s')
        ii_bins = 0
        num_bins = len(list(bin_groups.keys()))
        reporting_int = dlac._getReportingInt(reporting_percent, num_bins)
        #_warn('#############BIN NUMBER############### [[%d]] #############'%len(bin_groups))
        for (lower_group, upper_group), label in bin_groups.items():
            bin_N_sum = 0
            bin_width = 0
            bin_center = sum((lower_group, upper_group)) / 2.0
            bin_center_w = 0
            for ii in range(lower_group, upper_group+1):
                #_warn('for bin %d_%d ii:%d'%(lower_group, upper_group, ii))
                bin_width += 1
                bin_N_sum += groupIdToN.get(ii, 0)
                bin_center_w += groupIdToN.get(ii, 0) * ii
            bin_center_w = float(bin_center_w) / float(bin_N_sum)

            #_warn('number of users in range [%d, %d] is %d'%(lower_group, upper_group, bin_N_sum))
            
            # sql = 'SELECT group_id, feat, value, group_norm, N FROM %s where group_id >= %d AND group_id <= %d'%(self.featureTable, lower_group, upper_group)
            sql = 'SELECT group_id, feat, value, group_norm, std_dev FROM %s where group_id >= %d AND group_id <= %d'%(self.featureTable, lower_group, upper_group)
            groupFeatValueNorm = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            #pprint(groupFeatValueNorm)

            totalFeatCountForThisBin = float(0)
            featToValue = {}
            featToSummedNorm = {}
            for group_id, feat, value, norm, sd in groupFeatValueNorm:
            # for group_id, feat, value, norm, N in groupFeatValueNorm:
                if dlac.LOWERCASE_ONLY: feat = str(feat).lower()
                totalFeatCountForThisBin += value
                currentN = groupIdToN[group_id]
                try:
                    featToValue[feat] += value
                    featToSummedNorm[feat] += norm * currentN
                except KeyError:
                    featToValue[feat] = value
                    featToSummedNorm[feat] = norm * currentN

            #calculate mean and std_dev, using above info
            featToMeanNorm = {}
            featToSummedVar = {}
            for group_id, feat, _, norm, sd in groupFeatValueNorm:
                currentN = groupIdToN[group_id]
                meanNorm = featToSummedNorm[feat]/bin_N_sum
                try: 
                    featToSummedVar[feat] += currentN*((meanNorm - norm)**2 + (sd*sd))
                except KeyError:
                    featToSummedVar[feat] = currentN*((meanNorm - norm)**2 + (sd*sd))
                featToMeanNorm[feat] = meanNorm

            current_batch = [ ('_'.join(map(str,(lower_group, upper_group))),  k,  v, featToMeanNorm[k], sqrt(featToSummedVar[k] / bin_N_sum),
                               bin_N_sum, bin_center, bin_center_w, bin_width) for k, v in featToValue.items() ]
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, current_batch, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            # print 'N bin sum:', bin_N_sum
            # isql = 'INSERT INTO %s (group_id, feat, value, group_norm, N, bin_center, bin_center_w, bin_width) VALUES (%s)'%(newTable, '%s, %s, %s, %s, %s, %s, %s, %s')
            ii_bins += 1
            dlac._report('group_id bins', ii_bins, reporting_int, num_bins)

        mm.enableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        dlac.warn('Done creating new group_id-binned feature table.')

        outputdata = mm.executeGetList(self.corpdb, self.dbCursor, 'select group_id, N from `%s` group by group_id'%(newTable,), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        pprint(outputdata)

        # mm.execute(self.corpdb, self.dbCursor, 'drop table if exists `%s`'%(newTable,))
        return newTable

    def createTableWithRemovedFeats(self, p, minimumFeatSum = 0, groupFreqThresh = 0, setGFTWarning = False):
        """creates a new table with features that appear in more than p*|correl_field| rows, only considering groups above groupfreqthresh"""
        if not setGFTWarning:
            dlac.warn("""group_freq_thresh is set to %s. Be aware that groups might be removed during this process and your group norms will reflect this.""" % (groupFreqThresh), attention=True)
        toKeep, threshold = self._getKeepSet(p, minimumFeatSum, groupFreqThresh)
        if p in dlac.AUTO_POCC_MODES:
            p = threshold #auto tables are named by their group count, reproducible with --set_p_occ <count>
        label = str(p).replace('.', '_')
        if 'e' in label:
            label = label.replace('-','n')
        if minimumFeatSum > 1: 
            label += '_'+str(minimumFeatSum)
        return self.createNewTableWithGivenFeats(toKeep, label)

    def createNewTableWithGivenFeats(self, toKeep, label, featNorm=False):
        """Creates a new table only containing the given features"""
        featureTable = self.featureTable
        numToKeep = len(toKeep)
        newTable = featureTable+'$'+label
        dropTable = self.qb.create_drop_query(newTable)
        dropTable.execute_query()
        dlac.warn(" %s <new table %s will have %d distinct features.>" %(featureTable, newTable, numToKeep))
        createTable = self.qb.create_createTable_query(newTable).like(featureTable)
        createTable.execute_query()
        self.data_engine.standardizeTable(newTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)
        self.data_engine.disable_table_keys(newTable)
  
        num_at_time = 2000
        total = 0
        toWrite = []
        
        wsql = self.qb.create_insert_query(newTable).set_values([("group_id", ""), ("feat", ""), ("value", ""), ("group_norm", ""), ("feat_norm", "")]) if featNorm else self.qb.create_insert_query(newTable).set_values([("group_id", ""), ("feat", ""), ("value", ""), ("group_norm", "")])

        #iterate through each row, deciding whether to keep or not
        for featRow in self.getFeatAllSS(featNorm=featNorm):
            #print "%d %d" % (len(featRow), len(toWrite))
            if self.use_unicode and str(featRow[1]).lower() in toKeep:
                toWrite.append(featRow)
            elif not self.use_unicode and featRow[1].lower() in toKeep:
                toWrite.append(featRow)
            if len(toWrite) > num_at_time:
            #write those past the filter to the table
                wsql.execute_query(toWrite)
                total+= num_at_time
                if total % 100000 == 0: dlac.warn("%.1fm feature instances written" % (total/float(1000000)))
                toWrite = []

        #catch rest:
        if len(toWrite) > 0:
            #write those past the filter to the table
            wsql.execute_query(toWrite)

        dlac.warn("Done inserting.\nEnabling keys.")
        self.data_engine.enable_table_keys(newTable)
        dlac.warn("done.")
        self.featureTable = newTable
        return newTable

    def _getKeepSet(self, p, minimumFeatSum = 0, groupFreqThresh = 0):
        """creates a set of features occuring in less than p*|correl_field| rows"""
        #acquire the number of groups (need to base on corp table):
        featureTable = self.featureTable
        totalGroups = self.countGroups(groupFreqThresh)
        assert totalGroups > 0, 'NO GROUPS TO FILTER BASED ON (LIKELY group_freq_thresh IS TOO HIGH)'
        #acquire counts per feature (each row will come from a different correl_field)
        featCounts = self.getFeatureCounts(groupFreqThresh) #tuples of: feat, count (number of groups feature appears with)

        if featCounts and min(count for (feat, count) in featCounts) >= totalGroups:
            dlac.warn("Every feature is used by every group, so these appear not to be frequency- or count-based features (e.g. embeddings), which the occurrence filter is designed for", attention=True)

        if p in dlac.AUTO_POCC_MODES:
            threshold = self._autoMinGroups(p, totalGroups, [count for (feat, count) in featCounts])
        else:
            threshold = p
            if p < 1: threshold = int(round(p*totalGroups))
        dlac.warn (" %s [threshold: %d]" %(featureTable, threshold))

        #apply filter:
        toKeep = set()
        i = 0
        for (feat, count) in featCounts:
            if count >= threshold:
                if self.use_unicode:
                    toKeep.add(str(feat).lower())
                else:
                    toKeep.add(feat.lower())
            i += 1
            
            if (i % 1000000) == 0: print("    checked %d features" % i)
        
        #apply secondary filter
        if minimumFeatSum > 1:
            featSums = self.getFeatureValueSums()
            for (feat, fsum) in featSums:
                if self.use_unicode:
                    feat = str(feat).lower()
                else:
                    feat = feat.lower()
                if feat in toKeep:
                    if fsum < minimumFeatSum:
                        toKeep.remove(feat)
            
        return toKeep, threshold

    def _autoMinGroups(self, mode, N, counts):
        """Returns the minimum number of groups (of N) a feature must be used by for --set_p_occ auto or auto_prediction

        auto requires k = max(5, 0.75*sqrt(N)) groups and keeps at most 5*N^0.75 features, which suits DLA,
        where every feature must pass a multiple-test correction. auto_prediction halves k and keeps at most
        10*N features: features need not be significant on their own to help a prediction model.
        The cap is met by raising k, at most to N.
        """
        k = max(dlac.DEF_AUTO_POCC_K_MIN, int(math.ceil(dlac.DEF_AUTO_POCC_K_C * math.sqrt(N))))
        cap = int(dlac.DEF_AUTO_POCC_CAP_C * N ** dlac.DEF_AUTO_POCC_CAP_EXP)
        if mode == 'auto_prediction':
            k, cap = int(math.ceil(k / 2.0)), dlac.DEF_AUTO_PRED_POCC_CAP_C * N
        counts = sorted(counts, reverse=True)
        threshold = min(max(k, counts[cap] + 1), N) if len(counts) > cap else k
        numKept = sum(1 for c in counts if c >= threshold)
        dlac.warn(" --set_p_occ %s: %d groups; minimum groups per feature %d, at most %d features -> threshold %d groups (p_occ %.4f), keeping %d of %d features"
                  % (mode, N, k, cap, threshold, threshold / float(N), numKept, len(counts)))
        if mode == 'auto_prediction' and numKept > dlac.DEF_AUTO_PRED_POCC_WARN:
            dlac.warn("--set_p_occ auto_prediction keeps %d features; consider --set_p_occ auto or a fixed p_occ if models run out of memory"
                      % numKept, attention=True)
        return threshold

    def createTableWithReliableFeats(self, minReliability = dlac.DEF_RELIABILITY, groupFreqThresh = 0, splits = dlac.DEF_RELIABILITY_SPLITS, seed = dlac.DEFAULT_RANDOM_SEED):
        """creates a new table with the features of an n-gram or lexicon table whose split-half reliability across groups is at
        least minReliability (--feat_reliability_filter); see getFeatReliabilities for how reliability is calculated.

        Parameters
        ----------
        minReliability : :obj:`float`, optional
            keep features with at least this reliability
        groupFreqThresh : :obj:`int`, optional
            only groups with at least this many words
        splits : :obj:`int`, optional
            number of random split-halves to start with (getFeatReliabilities)
        seed : :obj:`int`, optional
            random seed for the splits (each group's split depends only on the seed and its id)

        Returns
        -------
        str
            the new table: the feature table name with $rel<minReliability> appended
        """
        rel = self.getFeatReliabilities(groupFreqThresh, splits, seed)
        return self._keepReliable(rel['feats'], rel['reliability'], minReliability, rel['numGroups'])

    def getFeatReliabilities(self, groupFreqThresh = 0, splits = dlac.DEF_RELIABILITY_SPLITS, seed = dlac.DEFAULT_RANDOM_SEED,
                             maxSplits = dlac.RELIABILITY_MAX_SPLITS, maxSE = dlac.RELIABILITY_MAX_SE):
        """each feature's reliability across groups, for an n-gram or lexicon table (--feat_reliability_filter,
        --create_reliability_table).

        Each group's messages are split at random into two halves with about as many words each
        (MessageTransformer.addSplitHalfMessageTables), the table's features are extracted from each half the way the
        table was made (--add_ngrams, or --add_lex_table on the half's 1grams; see _featTableRecipe), and each feature's
        group_norm in one half is correlated, across groups, with that in the other and Spearman-Brown corrected
        (splitHalfReliability) to the reliability of the whole group's group_norm. This is averaged over random splits.
        How much a feature's reliability varies from split to split is close to (1 - rel^2) / sqrt(groups) (as for a
        correlation over that many groups), so without splits, the first number of splits is the fewest that keep that
        standard error of the average below maxSE even when rel is 0: ceil(1 / (maxSE^2 * groups)), from 1 (100 or more
        groups split, for maxSE .1) to maxSplits, or 1 when most groups have only 2 messages (whose splits are all the
        same). With 2 or more splits, one more is added at a time while the measured standard error (median over
        features of their sd over splits / sqrt(splits)) is above maxSE, up to maxSplits. It is the share of the
        feature's variance across groups that is not sampling noise (its correlations with outcomes are attenuated by
        sqrt(reliability)). Messages, not words, are the sampled units, so words used in bursts (several times in one
        message) are not counted as more reliable than they are.

        The halves' counts are checked against the table's for a sample of groups; if they differ, the table was not
        made this way and nothing is calculated. Groups with fewer than 2 messages are left out. When most groups have
        at most two messages (e.g. -c message_id, or user-days), messages cannot be split, or every split of a group is
        the same and its halves can be far from equal: an n-gram table's reliability is then calculated from the table
        under a negative binomial model, assuming words are independent (see nbReliability), with a warning; a lexicon
        table is refused.

        Parameters
        ----------
        groupFreqThresh : :obj:`int`, optional
            only groups with at least this many words
        splits : :obj:`int`, optional
            number of random split-halves to start with (None: from the number of groups, as above)
        seed : :obj:`int`, optional
            random seed for the splits (each group's split depends only on the seed and its id)
        maxSplits : :obj:`int`, optional
            most split-halves
        maxSE : :obj:`float`, optional
            add split-halves while the median standard error of the features' reliabilities is above this

        Returns
        -------
        dict
            feats (sorted list), reliability (array over feats; negative when the halves disagree more than chance),
            numGroups (groups the reliabilities are from), method ('split-half' or 'negative binomial'), featNIndex
            (index into ns of each feature's n-gram size; 0 for lexicon tables), ns (n-gram sizes; [1] for lexicon
            tables), groupTotals (group id -> array of the group's total n-grams of each size, for every group passing
            groupFreqThresh), harmonicN (array: harmonic mean over the groups the reliabilities are from of their total
            n-grams of each size), splits (split-halves averaged) and se (array over feats: standard error over them; nan
            with one split)
        """
        recipe = self._featTableRecipe()
        ns = recipe['ns']

        #1. features, and the groups to use
        feats = sorted(set(self.getDistinctFeatures()))
        featIndex = dict((f, i) for i, f in enumerate(feats))
        featNIndex = np.array([ns.index(len(f.split())) if len(f.split()) in ns else 0 for f in feats]) if recipe['kind'] == 'ngram' \
            else np.zeros(len(feats), dtype=int)
        groups = self.getDistinctGroupsFromFeatTable()
        if groupFreqThresh:
            wordCounts = self.getGroupWordCounts()
            groups = [g for g in groups if wordCounts.get(g, 0) >= groupFreqThresh]
        assert groups, 'NO GROUPS TO FILTER BASED ON (LIKELY group_freq_thresh IS TOO HIGH)'
        if self.correl_field == self.messageid_field:
            few = len(groups)
        else:
            sql = self.qb.create_select_query(self.corptable).set_fields([self.correl_field, "count(*)"]).group_by([self.correl_field])
            msgCounts = dict((str(g), c) for g, c in sql.execute_query())
            few = sum(1 for g in groups if msgCounts.get(str(g), 0) <= 2)
        if few > len(groups) / 2.0: #1 message can't be split; 2 split only one way, often into unequal halves
            perGroup = 'only 1 message' if self.correl_field == self.messageid_field else 'only 1 or 2 messages'
            if recipe['kind'] != 'ngram':
                dlac.warn("There was %s per %s for %d of %d groups: messages cannot be split well, so the reliability "
                          "of %s cannot be calculated (the negative binomial model works only for n-gram tables)"
                          % (perGroup, self.correl_field, few, len(groups), self.featureTable), attention=True)
                sys.exit(1)
            dlac.warn("Because there was %s per %s for %d of %d groups, the reliability was parametrically "
                      "calculated using a Negative Binomial model and assuming independence of words (leading to slight "
                      "overestimate of reliability for common words)"
                      % (perGroup, self.correl_field, few, len(groups)), attention=True)
            reliability, totals = self._nbReliabilities(feats, featNIndex, ns, groups)
            withGrams = totals > 0 #groups with no n-grams of a size have no rate
            harmonicN = np.array([1.0 / (1.0 / t[w]).mean() if w.any() else 0.0 for t, w in zip(totals, withGrams)])
            return dict(feats=feats, reliability=reliability, numGroups=len(groups), method='negative binomial', featNIndex=featNIndex,
                        ns=ns, groupTotals=dict((g, totals[:, i]) for i, g in enumerate(groups)), harmonicN=harmonicN,
                        splits=0, se=np.zeros(len(feats)))
        if not splits: #enough splits for a standard error below maxSE (median message count: of groups that can be split)
            several = sorted(msgCounts[str(g)] for g in groups if msgCounts.get(str(g), 0) >= 2)
            splits = 1 if several[len(several) // 2] <= 2 else int(min(maxSplits, max(1, math.ceil(1.0 / (maxSE ** 2 * len(several))))))
        dlac.warn("Reliability: %d features over %d groups of %s.%s; extracting them from random split-halves of each "
                  "group's messages (%d splits%s)" % (len(feats), len(groups), self.corptable, self.correl_field, splits,
                                                      ', up to %d' % maxSplits if 1 < splits < maxSplits else ''))

        #2. split-halves: write the halves' message tables, extract the features from each, correlate
        from .messageTransformer import MessageTransformer
        mt = MessageTransformer(self.db_type, self.corpdb, self.corptable, self.correl_field, self.mysql_config_file,
                                self.message_field, self.messageid_field, self.encoding, self.use_unicode, self.lexicondb)
        rels, used, toDrop = [], [], []
        seconds = OrderedDict((step, 0.0) for step in ('splitting messages', 'extracting features', 'checking halves', 'correlating halves', 'dropping tables'))
        def timed(step, func, *args):
            start = time.time()
            result = func(*args)
            seconds[step] += time.time() - start
            return result
        try:
            while True:
                batch = list(range(len(rels), len(rels) + (splits if not rels else 1)))
                names, used = timed('splitting messages', mt.addSplitHalfMessageTables, groups, batch, seed)
                toDrop = [name for pair in names for name in pair]
                for halfA, halfB in names: #extract, correlate, and drop each split's tables before the next
                    made = []
                    try:
                        tablesA = timed('extracting features', self._extractLike, recipe, halfA, made)
                        tablesB = timed('extracting features', self._extractLike, recipe, halfB, made)
                        if not rels:
                            timed('checking halves', self._checkHalves, recipe, tablesA, tablesB, used[:dlac.RELIABILITY_CHECK_GROUPS], featIndex)
                        rels.append(timed('correlating halves', self.splitHalfReliability, tablesA, tablesB, used, feats))
                    finally:
                        timed('dropping tables', self._dropTables, made + [halfA, halfB])
                assert len(used) > 2, 'TOO FEW GROUPS WITH SEVERAL MESSAGES FOR --feat_reliability_filter'
                if len(rels) < 2: #one split: no measured standard error
                    se = np.full(len(feats), np.nan)
                    break
                se = np.std(rels, axis=0, ddof=1) / np.sqrt(len(rels))
                dlac.warn(" %d split-halves: standard error of reliabilities over features, 10%%/25%%/50%%/75%%/90%% = %s"
                          % (len(rels), ' / '.join('%.3f' % q for q in np.percentile(se, [10, 25, 50, 75, 90]))))
                if len(rels) >= maxSplits or np.median(se) <= maxSE:
                    break
        finally:
            self._dropTables(toDrop)
        dlac.warn(" %d split-halves took %s" % (len(rels), ', '.join('%s %.0fs' % (step, t) for step, t in seconds.items())))

        #3. reliability: averaged over the splits; groups' totals for per-group reliabilities
        reliability = np.mean(rels, axis=0)
        totals = self._groupNgramTotals(ns, groups, recipe['lowercase'])
        usedIndex = [i for i, g in enumerate(groups) if str(g) in set(str(u) for u in used)]
        harmonicN = np.zeros(len(ns))
        for j in range(len(ns)):
            t = totals[j, usedIndex]
            if (t > 0).any(): harmonicN[j] = 1.0 / (1.0 / t[t > 0]).mean()
        return dict(feats=feats, reliability=reliability, numGroups=len(used), method='split-half', featNIndex=featNIndex,
                    ns=ns, groupTotals=dict((g, totals[:, i]) for i, g in enumerate(groups)), harmonicN=harmonicN,
                    splits=len(rels), se=se)

    def _featTableRecipe(self):
        """how the feature table was made, from its name, for extracting its features from split-halves of the
        messages (getFeatReliabilities): {'kind': 'ngram', 'ns': [n, ...], 'lowercase': bool} for --add_ngrams tables
        (feat$1gram..., feat$1to3gram...), {'kind': 'lex', 'lexicon': name, 'weighted': bool, 'ns': [1], 'lowercase':
        True} for --add_lex_table tables (feat$cat_LEX[_w]$...$1gra), either optionally followed by --feat_occ_filter or
        --feat_reliability_filter suffixes ($0_05, $46, $rel0_3), which only remove features. Exits otherwise."""
        pieces = self.featureTable.split('$')
        def deny(why):
            dlac.warn("Unsure how %s was created (%s), so its features cannot be extracted from split-halves of the "
                      "messages: reliabilities work on tables made from -t and -c by --add_ngrams or --add_lex_table (on "
                      "the 1gram table), optionally filtered by --feat_occ_filter or --feat_reliability_filter"
                      % (self.featureTable, why), attention=True)
            sys.exit(1)
        if len(pieces) < 4 or pieces[0] != 'feat':
            deny('not a feat$ table name')
        if pieces[2] != self.corptable or pieces[3] != self.correl_field:
            deny('made from %s grouped by %s, not -t %s -c %s' % (pieces[2], pieces[3], self.corptable, self.correl_field))
        featType, extras = pieces[1], pieces[4:]
        isFilter = lambda e: re.match(r'^(\d+|0_\d+|rel\d+_\d+)$', e) is not None
        nMatch = re.match(r'^(\d+)(?:to(\d+))?gram(Up)?$', featType)
        if nMatch:
            if not all(isFilter(e) for e in extras):
                deny('suffix %s' % '$'.join(e for e in extras if not isFilter(e)))
            return dict(kind='ngram', ns=list(range(int(nMatch.group(1)), int(nMatch.group(2) or nMatch.group(1)) + 1)),
                        lowercase=not nMatch.group(3))
        if featType.startswith('cat_'):
            if not extras or extras[0] != '1gra' or not all(isFilter(e) for e in extras[1:]):
                deny('a lexicon table is expected to end in $1gra (from the 1gram table), then only filter suffixes')
            lex = featType[len('cat_'):]
            if re.search(r'_16to\d+$|_lw$', lex):
                deny('value-transformed or lexicon-normalized lexicon features')
            if lex.endswith('_w') and self._lexiconExists(lex[:-2]):
                return dict(kind='lex', lexicon=lex[:-2], weighted=True, ns=[1], lowercase=True)
            if self._lexiconExists(lex):
                return dict(kind='lex', lexicon=lex, weighted=False, ns=[1], lowercase=True)
            deny('no lexicon %s in %s' % (lex, self.lexicondb))
        deny('feature type %s' % featType)

    def _lexiconExists(self, name):
        from .lexicainterface.lexInterface import WeightedLexicon
        import os
        lexdb = self.lexicondb
        if self.db_type == 'sqlite' and not os.path.isabs(lexdb): #as load_lexicon finds it
            lexdb = os.path.join("/content" if os.path.exists("/content") else os.path.expanduser('~'), "sqlite_data", lexdb)
        lexicon = WeightedLexicon(lexicon_db=lexdb, mysql_config_file=self.mysql_config_file, lex_db_type=self.db_type,
                                  encoding=self.encoding, use_unicode=self.use_unicode)
        return lexicon.engine.tableExists(name)

    def _extractLike(self, recipe, messageTable, made):
        """extracts the features of a _featTableRecipe from messageTable with --add_ngrams' or --add_lex_table's
        methods; returns the tables to read (one per n-gram size, or the lexicon table) and adds every table it made
        to made"""
        from .featureExtractor import FeatureExtractor
        fe = FeatureExtractor(self.db_type, self.corpdb, messageTable, self.correl_field, self.mysql_config_file,
                              self.message_field, self.messageid_field, self.encoding, self.use_unicode, self.lexicondb)
        if recipe['kind'] == 'ngram':
            tables = []
            for n in recipe['ns']:
                tables.append(fe.addNGramTable(n, lowercase_only=recipe['lowercase'], metaFeatures=False))
                made.append(tables[-1])
            return tables
        made.append(fe.addNGramTable(1, metaFeatures=False)) #the word table the lexicon is applied to
        made.append(fe.addLexiconFeat(recipe['lexicon'], isWeighted=recipe['weighted']))
        return [made[-1]]

    def _dropTables(self, tables):
        for table in tables:
            self.qb.create_drop_query(table).execute_query()

    def _readTables(self, tables, groups, featIndex, where=''):
        """(group index, feature index, value, group_norm) of tables' rows for groups and features in featIndex"""
        groupIndex = dict((str(g), i) for i, g in enumerate(groups))
        for table in tables:
            sql = self.qb.create_select_query(table).set_fields(["group_id", "feat", "value", "group_norm"])
            if where: sql = sql.where(where)
            for gid, feat, value, gn in self.data_engine.execute_get_SSCursor(sql.toString()):
                gi, fi = groupIndex.get(str(gid)), featIndex.get(feat)
                if gi is not None and fi is not None:
                    yield gi, fi, float(value), float(gn)

    def _checkHalves(self, recipe, tablesA, tablesB, groups, featIndex):
        """exits unless, for groups, the halves' values add up to the feature table's (so the table was made as
        _featTableRecipe says): exactly for counts, within rounding for a weighted lexicon's (integer) values"""
        where = "group_id IN ('%s')" % "','".join(str(g) for g in groups)
        full, halves = dict(), dict()
        for gi, fi, value, gn in self._readTables([self.featureTable], groups, featIndex, where):
            full[(gi, fi)] = value
        for gi, fi, value, gn in self._readTables(tablesA + tablesB, groups, featIndex, where):
            halves[(gi, fi)] = halves.get((gi, fi), 0.0) + value
        tolerance = 1.5 if recipe.get('weighted') else 1e-6
        keys = set(full) | set(halves)
        mismatches = sum(1 for k in keys if abs(full.get(k, 0.0) - halves.get(k, 0.0)) > tolerance)
        if not keys or mismatches > dlac.RELIABILITY_CHECK_TOLERANCE * len(keys):
            dlac.warn("Unsure how %s was created: features extracted from split-halves of %s's messages do not add up to "
                      "its values in %d of %d (group, feature) pairs of %d groups (was it made from this message table "
                      "and group field, from its unfiltered 1grams, without collocations or a tokenized table?)"
                      % (self.featureTable, self.corptable, mismatches, len(keys), len(groups)), attention=True)
            sys.exit(1)

    def splitHalfReliability(self, tablesA, tablesB, groups, feats):
        """each feature's split-half reliability across groups: the correlation over groups between its group_norm in
        feature tables extracted from one half of each group's messages (tablesA, one per n-gram size or a lexicon
        table) and from the other half (tablesB), Spearman-Brown corrected to the whole (splitHalfFromSums). Groups
        without the feature in a half count as 0.

        Returns
        -------
        numpy array
            reliability of each of feats
        """
        featIndex = dict((f, i) for i, f in enumerate(feats))
        halves = []
        for tables in (tablesA, tablesB):
            rows, cols, data = [], [], []
            for gi, fi, value, gn in self._readTables(tables, groups, featIndex):
                rows.append(gi); cols.append(fi); data.append(gn)
            halves.append(csr_array((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                                    shape=(len(groups), len(feats))))
        A, B = halves
        colSum = lambda M: np.asarray(M.sum(axis=0)).ravel()
        return splitHalfFromSums(colSum(A), colSum(B), colSum(A.multiply(A)), colSum(B.multiply(B)), colSum(A.multiply(B)), len(groups))

    def _groupNgramTotals(self, ns, groups, lowercase = True):
        """(len(ns) x len(groups)) array of each group's total n-grams of each size, from the feat$meta_ tables, else
        (with a warning) the groups' word counts for every size"""
        pieces = self.featureTable.split('$')
        groupIndex = dict((str(g), i) for i, g in enumerate(groups))
        totals = np.zeros((len(ns), len(groups)))
        for j, n in enumerate(ns):
            metaTable = '$'.join(['feat', 'meta_%dgram%s' % (n, '' if lowercase else 'Up')] + pieces[2:4])
            if self.data_engine.tableExists(metaTable):
                sql = self.qb.create_select_query(metaTable).set_fields(['group_id', 'value']).where("feat = '_total%dgrams'" % n)
                for g, v in sql.execute_query():
                    if str(g) in groupIndex: totals[j, groupIndex[str(g)]] = float(v)
        if not totals.any(axis=1).all():
            dlac.warn("No meta tables with n-gram totals for %s: taking each group's word count as its total n-grams" % self.featureTable)
            wordCounts = dict((str(g), c) for g, c in self.getGroupWordCounts().items())
            totals[:] = [[float(wordCounts.get(str(g), 0)) for g in groups]] * len(ns)
        return totals

    def _keepReliable(self, feats, reliability, minReliability, numGroups):
        """writes the table of features with at least minReliability (createTableWithReliableFeats)"""
        toKeep = set((str(f) if self.use_unicode else f).lower() for f, rel in zip(feats, reliability) if rel >= minReliability)
        qs = np.percentile(reliability, [10, 25, 50, 75, 90])
        dlac.warn(" %s [reliability over %d groups: 10%%/25%%/50%%/75%%/90%% = %s; keeping %d of %d features with reliability >= %s]"
                  % (self.featureTable, numGroups, ' / '.join('%.2f' % q for q in qs), len(toKeep), len(feats), minReliability))
        return self.createNewTableWithGivenFeats(toKeep, 'rel' + str(minReliability).replace('.', '_'))

    def createReliabilityTable(self, groupFreqThresh = 0, splits = dlac.DEF_RELIABILITY_SPLITS, seed = dlac.DEFAULT_RANDOM_SEED):
        """writes the reliabilities of an n-gram or lexicon table's features (--create_reliability_table) to a table named like it
        with relf$ in place of feat$ and how they were calculated appended: $sh<k> when from k random split-halves of
        the groups' messages, $nb when from the negative binomial model (relf$1gram$msgs$user_id$sh1), with a row for
        each of the feature table's rows (group, feature) whose group has at least groupFreqThresh words:
        value is the feature's reliability across groups (getFeatReliabilities; the same in each of its rows, and
        negative when its halves agree less than chance), and group_norm is the reliability of that group's group_norm
        of the feature (groupReliability: the across-group reliability projected to the group's number of n-grams, or
        words for a lexicon table). Groups where a feature does not occur get no row, as in the feature table; their reliability follows
        from value and the group's total (feat$meta_ tables) by groupReliability.

        Returns
        -------
        str
            the new table's name
        """
        featureTable = self.featureTable
        pieces = featureTable.split('$')
        newTable = '$'.join(['relf'] + pieces[1:])
        if self.db_type == 'mysql' and len(newTable) + len('$sh6') > 64:
            dlac.warn("%s$sh6 is longer than MySQL's 64 characters; use a shorter message table name" % newTable, attention=True)
            sys.exit(1)
        rel = self.getFeatReliabilities(groupFreqThresh, splits, seed)
        newTable += '$nb' if rel['method'] == 'negative binomial' else '$sh%d' % rel['splits']
        featIndex = dict((f, i) for i, f in enumerate(rel['feats']))
        reliability, featNIndex, harmonicN = rel['reliability'], rel['featNIndex'], rel['harmonicN']
        groupTotals = dict((str(g), t) for g, t in rel['groupTotals'].items())
        qs = np.percentile(reliability, [10, 25, 50, 75, 90])
        dlac.warn(" %s [%s reliability over %d groups: 10%%/25%%/50%%/75%%/90%% = %s; harmonic mean n-grams per group: %s]"
                  % (featureTable, rel['method'], rel['numGroups'], ' / '.join('%.2f' % q for q in qs),
                     ', '.join('%.0f' % h for h in harmonicN)))

        self.qb.create_drop_query(newTable).execute_query()
        self.qb.create_createTable_query(newTable).like(featureTable).execute_query()
        if self.db_type == 'mysql': #value holds a reliability, not a count
            self.data_engine.execute("ALTER TABLE `%s` MODIFY value DOUBLE" % newTable)
        self.data_engine.standardizeTable(newTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)
        self.data_engine.disable_table_keys(newTable)
        wsql = self.qb.create_insert_query(newTable).set_values([("group_id", ""), ("feat", ""), ("value", ""), ("group_norm", "")])
        dlac.warn("Writing %s" % newTable)
        toWrite, written = [], 0
        for (gid, feat, value, gn) in self.getFeatAllSS(featNorm=False):
            totals, fi = groupTotals.get(str(gid)), featIndex.get(feat)
            if totals is None or fi is None: #group below group_freq_thresh
                continue
            n = featNIndex[fi]
            toWrite.append((gid, feat, float(reliability[fi]), float(groupReliability(reliability[fi], totals[n], harmonicN[n]))))
            if len(toWrite) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                wsql.execute_query(toWrite)
                written += len(toWrite)
                toWrite = []
                if written % 1000000 < dlac.MYSQL_BATCH_INSERT_SIZE: dlac.warn("  %.1fm rows written" % (written / 1e6))
        if toWrite:
            wsql.execute_query(toWrite)
            written += len(toWrite)
        dlac.warn("Done: %d rows. Enabling keys." % written)
        self.data_engine.enable_table_keys(newTable)
        return newTable

    def _nbReliabilities(self, feats, featNIndex, ns, groups):
        """each feature's reliability across groups under a negative binomial model (nbReliability), from the
        feature table and each group's total n-grams (meta tables, else value / group_norm)"""
        featureTable = self.featureTable
        pieces = featureTable.split('$')
        groupIndex = dict((str(g), i) for i, g in enumerate(groups))
        totals = np.zeros((len(ns), len(groups)))
        for j, n in enumerate(ns):
            metaTable = '$'.join(['feat', 'meta_' + (pieces[1] if len(ns) == 1 else '%dgram' % n)] + pieces[2:4])
            if self.data_engine.tableExists(metaTable):
                sql = self.qb.create_select_query(metaTable).set_fields(['group_id', 'value']).where("feat = '_total%dgrams'" % n)
                for g, v in sql.execute_query():
                    if str(g) in groupIndex: totals[j, groupIndex[str(g)]] = float(v)
        inferTotals = not totals.any(axis=1).all()
        if inferTotals:
            dlac.warn("No meta table with n-gram totals for %s: taking each group's total as value / group_norm" % featureTable)
            totals[:] = 0
        F = len(feats)
        featIndex = dict((f, i) for i, f in enumerate(feats))
        sumC, sumC2, sumCN = np.zeros(F), np.zeros(F), np.zeros(F)
        rows = []
        for (gid, feat, value, gn) in self.getFeatAllSS(featNorm=False):
            gi, fi = groupIndex.get(str(gid)), featIndex.get(feat)
            if gi is None or fi is None:
                continue
            if inferTotals and gn:
                totals[featNIndex[fi], gi] = max(totals[featNIndex[fi], gi], float(value) / float(gn))
            rows.append((fi, gi, float(value)))
        for fi, gi, c in rows:
            sumC[fi] += c
            sumC2[fi] += c * c
            sumCN[fi] += c * totals[featNIndex[fi], gi]
        reliability = np.zeros(F)
        for j in range(len(ns)):
            fs = np.nonzero(featNIndex == j)[0]
            if not len(fs):
                continue
            withGrams = totals[j] > 0 #groups with no n-grams of this size have no rate
            reliability[fs] = nbReliability(sumC[fs], sumC2[fs], sumCN[fs], totals[j].sum(), (totals[j] ** 2).sum(),
                                            (1.0 / totals[j][withGrams]).mean() if withGrams.any() else np.inf)
        return reliability, totals

    def createStandardizedFeatTable(self, use_mean = True, use_std = True, groupFreqThresh = 0, setGFTWarning = False, where=None):
        #creates a copy of a feature table where all group_norms have been standardized
        #variables effecting standardization: GFT
        #use_mean : whether to mean center the number of words in the ngrams
        #use_std : whether to divide by standard deviation
        featTable = self.featureTable
        dlac.warn("""Standardizing %s \n -- note that if the table was sparse with many columns then this will make it very large unless use_mean is set to false.""" % (str(featTable)), attention=True)


        #1. apply GFT
        groups = []
        if not setGFTWarning:
            dlac.warn("""group_freq_thresh is set to %s. Be aware that groups might be used during mean and std calculation for standardizing.""" % (groupFreqThresh), attention=True)
        if groupFreqThresh:
            groupCnts = self.getGroupWordCounts(where)
            for group, wordCount in groupCnts.items():
                if (wordCount >= groupFreqThresh):
                    groups.append(group)
        else:
            groups = self.getDistinctGroups(where)

        #2. Get Sparse X:
        (groupNorms, featureNames) = self.getGroupNormsSparseFeatsFirst(groups)
        gnList = [groupNorms[feat] for feat in featureNames] #list of dictionaries of group_id => group_norm
        #featToIndex = dict([(featureNames[i], i) for i in range(len(featureNames))])
        groupToIndex = dict([(groups[i], i) for i in range(len(groups))])
        row = []
        col = []
        data = []
        # columns: features.
        for featIndex in range(len(gnList)):
            colData = gnList[featIndex]
            for groupid, value in colData.items():
                    row.append(groupToIndex[groupid])
                    col.append(featIndex)
                    data.append(value)
        assert all([isinstance(x,numbers.Number) for x in data]), "Data is corrupt, there are non float elements in the group norms (some might be NULL?)"
        X = csr_array((data,(row,col)), shape = (len(groups), len(featureNames)), dtype=np.float64)
        if use_mean: 
            X = X.todense()
        dlac.warn("\n X.shape: %s]" % str(X.shape))

        #3. Apply STandardization: 
        scaler = StandardScaler(with_mean = use_mean, with_std=use_std)
        dlac.warn("\n [Applying StandardScaler to X: %s]" % str(scaler))
        X = scaler.fit_transform(X)
        dlac.warn(" [Done]\n")
        
        #4. Create new table:
        featTable = self.featureTable
        if isinstance(featTable,list):
            dlac.warn("Multiple  feature tables; only running on first: %s"%str(featTable))
            featTable = featTable[0]
        tableName = featTable+'$z'
        if not use_mean: tableName+='NoMe'
        if not use_std: tableName+='NoStd'
        columns = mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, featTable, mysql_config_file=self.mysql_config_file)
        if not columns: raise ValueError("One of your feature tables probably doesn't exist")
        currentType = columns['feat']
        intGrabber = re.compile(r'\d+')
        featNameSize = int(intGrabber.search(currentType).group())
        featNameGrabber = re.compile(r'^feat\$([^\$]+)\$')
        featureType = featNameGrabber.match(featTable).group(1)
        
        tableName = self.createFeatureTable(featureType, "VARCHAR(%d)"%featNameSize, 'DOUBLE', tableName)


        #5. Insert into table
        if X.shape[0]*X.shape[1] < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
        for row in range(X.shape[0]):
            cf_id = groups[row]

            wsql = """INSERT INTO """+tableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
            if self.use_unicode:
                rows = [(featureNames[col], X[row][col], X[row][col]) for col in range(X.shape[1])]
            else:
                rows = [(featureNames[col].encode('utf-8'), X[row][col], X[row][col]) for col in range(X.shape[1])]
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        if X.shape[0]*X.shape[1] < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
        dlac.warn("Done\n")

        return tableName

    def createInterpolatedFeatTable(self, days = 1, dateField = 'created_at', minToImpute = 2, maxDaysUnitsEmpty = 4, groupFreqThresh = 0, setGFTWarning = False, where=None):
        #creates a new feature table at a higher level of aggregation
        #days: what units of time to aggregate/interpolate into
        #maxDayUnitsEmpty: maximum number of days missed to keep a user.
        #TODO: handle maxDaysUnitsEmpty

        featureTable = self.featureTable

        ## 0. Create new interpolated table:
        (_, name, corpTable, oldGroupField) = featureTable.split('$')[:4]
        assert oldGroupField == self.messageid_field, "Interpolate currently only works if interpolating message-level features"
        theRest = featureTable.split('$')[4:]
        feature_name = name[:11]+'_'+'_'.join([i[:3] for i in theRest])
        dayName = days
        if dayName%1.0 > 0: dayName = str(int(days*10)/10).replace('.', '_')
        else: dayName = str(int(dayName))
        newTable = 'feat$intrp'+ dayName \
                   + '_'+feature_name+'$'+corpTable+'$'+self.correl_field
        columns = mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, featureTable, mysql_config_file=self.mysql_config_file)
        if not columns: raise ValueError("One of your feature tables probably doesn't exist")
        currentType = columns['feat']
        columns = mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, self.corptable, mysql_config_file=self.mysql_config_file)
        if not columns: raise ValueError("One of your feature tables probably doesn't exist")
        groupidType = columns[self.correl_field]
        tableName = self.createFeatureTable(feature_name, currentType, 'DOUBLE', newTable)
        ##add extra columns:
        mm.execute(self.corpdb, self.dbCursor, "ALTER TABLE %s ADD time INT, ADD %s %s" %
                   (tableName, self.correl_field, groupidType), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
     
        ## we are appending a date to the correl_field so group_id in feat table must be varchar
        if 'int' in groupidType:
            mm.execute(self.corpdb, self.dbCursor, "ALTER TABLE %s MODIFY group_id VARCHAR(64)" %
                       (tableName), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
 
        dlac.warn("""Interpolating %s to the %s level.""" % (str(featureTable), self.correl_field))
        
        ## 1. apply GFT to get users we care about: 
        groups = []
        if not setGFTWarning:
            dlac.warn("""group_freq_thresh is set to %s. Only %s s meeting this will have interpolated features""" %
                      (groupFreqThresh, self.correl_field), attention=True)
        if groupFreqThresh:
            groupCnts = self.getGroupWordCounts(where)
            for group, wordCount in groupCnts.items():
                if (wordCount >= groupFreqThresh):
                    groups.append(group)
        else:
            groups = self.getDistinctGroups(where)
        dlac.warn("""  Interpolating for up to %d %s s.""" % (len(groups), self.correl_field))
        gList = "','".join([str(g) for g in groups])

        ## 2. Get the minimum and maximum dates:
        dlac.warn("""  [Finding good date range]""" )
        minMaxQuery = "SELECT max(min_date), min(max_date), min(min_date), max(max_date) FROM (SELECT %s, MIN(%s) as min_date, MAX(%s) as max_date FROM %s where %s in ('%s') group by %s) as a " % \
                                         (self.correl_field, dateField, dateField, self.corptable, self.correl_field, gList, self.correl_field)
        dlac.warn("MinMaxQuery: " + minMaxQuery)
        minIntersectDT, maxIntersectDT, minUnionDT, maxUnionDT  = mm.executeGetList(self.corpdb, self.dbCursor, minMaxQuery, mysql_config_file=self.mysql_config_file)[0]

        dtClosestToMin = mm.executeGetList(self.corpdb, self.dbCursor, "SELECT min(max_date) FROM (SELECT %s, MAX(%s) as max_date FROM %s where %s in ('%s') AND %s <= '%s' group by %s) as a " % \
                                         (self.correl_field, dateField, self.corptable, self.correl_field, gList, dateField, minIntersectDT, self.correl_field), mysql_config_file=self.mysql_config_file)[0][0]
        dtClosestToMax = mm.executeGetList(self.corpdb, self.dbCursor, "SELECT max(min_date) FROM (SELECT %s, MIN(%s) as min_date FROM %s where %s in ('%s') AND %s >= '%s' group by %s) as a " % \
                                         (self.correl_field, dateField, self.corptable, self.correl_field, gList, dateField, maxIntersectDT, self.correl_field), mysql_config_file=self.mysql_config_file)[0][0]
        if not isinstance(minIntersectDT, datetime.datetime):
            minIntersectDT = dtParse(minIntersectDT, ignoretz = True)
            maxIntersectDT = dtParse(maxIntersectDT, ignoretz = True)
            minUnionDT = dtParse(minUnionDT, ignoretz = True)
            maxUnionDT = dtParse(maxUnionDT, ignoretz = True)
            dtClosestToMin = dtParse(maxUnionDT, ignoretz = True)
            dtClosestToMax = dtParse(maxUnionDT, ignoretz = True)

            
        dayDiff = (maxIntersectDT - minIntersectDT).days #difference in days
        maxDiffPerUnit = int(dayDiff/days) #convert to timesteps
        assert dayDiff > 0, "\nDayDiff (%d) was negative or zero. Increase GFT. Interpolating %ss over %ddays: 0 = %s through %d = %s"% \
                  (dayDiff, self.correl_field, days, str(minIntersectDT.date()), maxDiffPerUnit, str(maxIntersectDT.date()))

        ## 3. Get Features x SubIds (in Sparse X form):
        oldFeatures = FeatureGetter(self.db_type, self.corpdb, self.corptable, oldGroupField, self.mysql_config_file, self.message_field, self.messageid_field, self.encoding, True, self.lexicondb, featureTable)
        dateWhere = "%s >= DATE('%s') AND %s <= DATE('%s')" % (dateField, dtClosestToMin.date(), dateField, dtClosestToMax.date()) #used when querying mids + dates
        groupsWhere = "group_id in (SELECT %s FROM %s WHERE "%(oldGroupField, self.corptable) +dateWhere+" AND %s in ('%s'))" %(self.correl_field, gList)
        #print("groupsWhere", groupsWhere)#debug
        (groupNormsByMid, featureNames) = oldFeatures.getGroupNormsSparseGroupsFirst(where = groupsWhere)
        dlac.warn("""  [done]""")
        
        ## 4. Create X (time steps) and Y (group_norm) per group: 
        lengroups = len(groups)
        dlac.warn("\nInterpolating %d %ss over %ddays: 0 = %s through %d = %s (full range: %s through %s)"% \
                  (lengroups, self.correl_field, days, str(minIntersectDT.date()), maxDiffPerUnit, \
                   str(maxIntersectDT.date()), str(dtClosestToMin.date()), str(dtClosestToMax.date())))
        gnum = 0
        wsql = """INSERT INTO """+newTable+""" (group_id, feat, value, group_norm, time, """+self.correl_field+""") values (%s, %s, %s, %s, %s, %s)"""
        for group in groups:

            #status tracker:
            if gnum%100==0:
                dlac.warn("   Read and considered interpolating %d %ss (%.2f complete) "%(gnum, self.correl_field, gnum/lengroups))
            gnum +=1

            ## 5. go throgh current users messages and align with feature table data:
            groupXYs = {f: [] for f in featureNames} #feat->[(X, y)...] where X is datediff from min and y is group_norm; adds zeros here
            midsAndDates = self.getMidAndExtraForCorrelField(group, dateField, where=dateWhere, warnMsg = False)
            if len(midsAndDates) < minToImpute:#make total messages is at least enough before bothering:
                dlac.warn(" !Only %d messages to impute for %s %s. Not enough. SKIPPING" % \
                          (len(midsAndDates), self.correl_field, str(group)))
                continue
            uniqueDtInts = set() #tracks total unique dts
            uniqueDtIntsInRange = set() #tracks total unique dts in the range
            for dateRow in midsAndDates:
                #print("dateRow", dateRow)#debug
                mid = dateRow[0]
                if mid in groupNormsByMid:
                    
                    ## 6. get difference in dates by number of days parameter:
                    #print("checking", mid)
                    dt = dateRow[1]
                    if not isinstance(dt, datetime.datetime):
                        dt = dtParse(dt, ignoretz = True)
                    dayDiff = (dt - minIntersectDT).days
                    diffPerUnit = int(dayDiff/days)
                    uniqueDtInts.add(diffPerUnit)
                    if diffPerUnit >= 0:
                        uniqueDtIntsInRange.add(diffPerUnit)

                    for feat in featureNames:
                        try:
                            groupXYs[feat].append((diffPerUnit, groupNormsByMid[mid][feat]))#appends (X, y) to entry for feat
                        except KeyError: #feat must be missing in group Norms = 0
                            groupXYs[feat].append((diffPerUnit, 0))
            if len(uniqueDtInts) < minToImpute:#check that there are enough unique dates: 
                dlac.warn(" !Only %d unique dates to impute for %s %s (need at least %d). SKIPPING"% \
                          (len(uniqueDtInts), self.correl_field, str(group), minToImpute))
                continue
            elif len(uniqueDtIntsInRange) < minToImpute-1:#check that there are enough unique dates: 
                dlac.warn(" !Only %d unique dates i the range %s %s (need at least %d). SKIPPING"% \
                          (len(uniqueDtIntsInRange), self.correl_field, str(group), minToImpute-1))
                continue

            ## 7. Interpolate (note: this and 8 could be parallelized)
            minX, maxX = min(uniqueDtInts), max(uniqueDtInts)
            if maxX - minX < maxDiffPerUnit:#left and/or right will be left out
                dlac.warn(" !Warning, %s %s has smaller range (%d to %d) than max (0 to %d)."% \
                          (self.correl_field, str(group), minX, maxX, maxDiffPerUnit))
            newX = list(range(minX, maxX+1)) #i.e. range to interpolate over
            newYs = dict()
            newYs["_%ddInter"%days] = [1.0]*len(newX) #intercept to make sure day counts even if zero
            #fit:
            for feat in featureNames:
                x, y = zip(*groupXYs[feat])
                #print(x, y)
                try: 
                    f = interp1d(x, y, 'slinear')
                    newYs[feat] = f(newX)
                except ValueError:
                    dlac.warn("THIS SHOULD HAVE BEEN CAUGHT EARLIER.\n Not enough values to impute for %s %s. SKIPPING"%\
                              (self.correl_field, str(group), feat))
                    #NOTE: this should only happen on the first feat; because zeros have been added
                    break

            ## 8. Write to feature table DB, including time and initial group_id (the version without time)
            #print(newX, newYs)#debug
            rows = []
            if not self.use_unicode:
                rows = [(str(group)+'_'+str(newX[i]), feat, newX[i], Ys[i], newX[i], group) for i in range(len(newX)) for feat, Ys in newYs.items() if Ys[i] != 0]
            else:
                rows = [((str(group)+'_'+str(newX[i])).encode('utf-8'), feat.encode('utf-8'), newX[i], \
                         Ys[i], newX[i], group) for i in range(len(newX)) for feat, Ys in newYs.items() if Ys[i] != 0]
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading, Interpolating, and Inserting into '%s'."% tableName )
        mm.execute(self.corpdb, self.dbCursor, "ALTER TABLE %s ADD INDEX (time), ADD INDEX (%s)" % (tableName, self.correl_field), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return tableName


    
    def addFeatNorms(self, ReCompute = True, groupFreqThresh = 0, setGFTWarning=True):
        """Adds the mean normalization by feature (z-score) for each feature"""
        ##TODO: add feat_norm column if doesn't exist
        where = None
        if not ReCompute: where = 'feat_norm is null'
        groupNorms = self.getGroupNorms(where = where) #contains group_id, feat, group_norm
        if not setGFTWarning:
            dlac.warn("""group_freq_thresh is set to %s. Be aware that groups might be removed during this process and your group norms will reflect this.""" % (groupFreqThresh), attention=True)

        fMeans = self.addFeatTableMeans(groupNorms = groupNorms, groupFreqThresh=groupFreqThresh) #mean, std, zero

        wsql = """UPDATE """+self.featureTable+""" SET feat_norm = %s where group_id = %s AND feat = %s"""
        featNorms = []
        num_at_time = 2000
        numWritten = 0
        for (group_id, feat, group_norm) in groupNorms:
            if dlac.LOWERCASE_ONLY: feat = feat.lower()
            if (feat):
                fn = ( ((group_norm - fMeans[feat][0]) / float(fMeans[feat][1]), group_id, feat) )
                featNorms.append(fn)
                if len(featNorms) >= num_at_time:
                    mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, featNorms, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                    featNorms = []
                    numWritten += num_at_time
                    if numWritten % 100000 == 0: dlac.warn("%.1fm feature instances updated out of %dm" % 
                                                        ((numWritten/float(1000000)), len(groupNorms)/1000000))
        
        #write values back in 
        if featNorms: 
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, featNorms, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return True

        
    def findMeans(self, field='group_norm', addZeros = True, groupNorms = None, groups = set(), groupFreqThresh = 0, where = None):
        """Finds feature means from group norms"""

        if groupFreqThresh:
            groupCnts = self.getGroupWordCounts(where)
            for group, wordCount in groupCnts.items():
                if (wordCount >= groupFreqThresh):
                    groups.add(group)
        
        if not groupNorms:
            groupNorms = self.getGroupNorms() #contains group_id, feat, group_norm

        #turn into dict of lists:
        fNumDict = dict() 
        for (gid, feat, gn) in groupNorms:
            if (len(groups) < 1) or (gid in groups): 
                if dlac.LOWERCASE_ONLY: feat = feat.lower()
                if feat: 
                    try:
                        fNumDict[feat].append(gn)
                    except KeyError:
                        fNumDict[feat] = [gn]                      

        #add zeros for groups missing featrues
        if addZeros:
            totalGroups = self.countGroups(groupFreqThresh)
            for feat, nums in fNumDict.items():
                difference = totalGroups - len(nums)
                nums.extend([0]*difference)
            
        #calculate values:
        fMeans = dict()
        for feat, nums in fNumDict.items():
            nums = array([float(d) for d in nums])
            (m, s) = (mean(nums), std(nums))
            z = (0 - m) / float(s)
            fMeans[feat] = (m, s, z)

        return fMeans


    def addFeatTableMeans(self, field='group_norm', groupNorms = None, groupFreqThresh = 0, groups=set()):
        """Add to the feature mean table: mean, standard deviation, and zero_mean for the current feature table"""
       
        #CREATE TABLE
        meanTable = 'mean$'+self.featureTable
        mm.execute(self.corpdb, self.dbCursor, "DROP TABLE IF EXISTS %s" % meanTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        featType = mm.executeGetList(self.corpdb, self.dbCursor, "SHOW COLUMNS FROM %s like 'feat'" % self.featureTable)[0][1]
        sql = """CREATE TABLE %s (feat %s, mean DOUBLE, std DOUBLE, zero_feat_norm DOUBLE, PRIMARY KEY (`feat`)) CHARACTER SET %s COLLATE %s ENGINE=%s""" % (meanTable, featType, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()], dlac.DEF_MYSQL_ENGINE)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        fMeans = self.findMeans(field, True, groupNorms, groupFreqThresh = groupFreqThresh)
        fMeansList = [(k, v[0], v[1], v[2]) for k, v in fMeans.items()]
        #print fMeansList #debug

        #WRITE TO TABLE:
        sql = """INSERT INTO """+meanTable+""" (feat, mean, std, zero_feat_norm) VALUES (%s, %s, %s, %s)"""
        mm.executeWriteMany(self.corpdb, self.dbCursor, sql, fMeansList, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return fMeans


    def createCollocRefinedFeatTable(self, threshold = 3.0, featNormTable=False):
        #n = the number of words in the ngrams
        #uses pmi to remove uncommon collocations:
        featureTable = self.featureTable
        dlac.warn(featureTable)
        wordGetter = self.getWordGetter()
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        jointFreqs = self.getSumValuesByFeat()
        wordFreqs = dict(wordGetter.getSumValuesByFeat())
        allFreqs = wordGetter.getSumValue()

        keepers = set()
        for (colloc, freq) in jointFreqs:
            # words = tokenizer.tokenize(colloc)
            # If words got truncated in the creation of 1grams, we need to account for that
            words = [word[:dlac.VARCHAR_WORD_LENGTH] for word in tokenizer.tokenize(colloc)]
            if (len(words) > 1):
                indFreqs = [wordFreqs[w] for w in words if w in wordFreqs]
                pmi = FeatureRefiner.pmi(freq, indFreqs, allFreqs, words = words)
                # print "%s: %.4f" % (colloc, pmi)#debug
                if pmi > (len(words)-1)*threshold: 
                    keepers.add(colloc)
            else:
                keepers.add(colloc)
        return self.createNewTableWithGivenFeats(keepers, "pmi%s"%str(threshold).replace('.', '_'), featNormTable)

    def getCollocsWithPMI(self):
        '''
        :inputs: self.featureTable
        calculates PMI for each ngram that is >1
        :returns: a dict of colloc => [pmi, num_tokens, pmi_threshold_val]
            **pmi_threshold_val is pmi/(num_tokens-1), thats what --feat_colloc_filter is based on
        '''
        featureTable = self.featureTable
        dlac.warn(featureTable)
        wordGetter = self.getWordGetter()
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        jointFreqs = self.getSumValuesByFeat()
        wordFreqs = dict(wordGetter.getSumValuesByFeat())
        allFreqs = wordGetter.getSumValue()

        keepers = set()
        collocPMIs = {}
        count = 0
        print("len(jointFreqs): " + str(len(jointFreqs)))
        for (colloc, freq) in jointFreqs:
            count +=1
            if count % 50000 == 0:
                print("calculating pmi for {}th feature".format(count))
            words = [word[:dlac.VARCHAR_WORD_LENGTH] for word in tokenizer.tokenize(colloc)]
            if (len(words) > 1):
                indFreqs = [wordFreqs[w] for w in words if w in wordFreqs]
                pmi = FeatureRefiner.pmi(freq, indFreqs, allFreqs, words = words)
                collocPMIs[colloc] =[colloc, freq, pmi, len(words), pmi/(len(words)-1)]
        return collocPMIs

    @staticmethod
    def pmi(jointFreq, indFreqs, allFreq, words = None):
        #  log p(w1, w2) / (p(w1)p(w2)) 
        allFreq = float(allFreq) #to insure floating point div.
        jointP = float(jointFreq) / allFreq
        
        denominator = 1.0
        for iFreq in indFreqs:
            denominator *= (float(iFreq) / allFreq)
        out = 0.0
        try:
            out = log2( jointP/denominator)
        except ZeroDivisionError:
            print(jointFreq, indFreqs, allFreq, words)
        return out

    @staticmethod
    def salience(jointFreq, indFreqs, allFreq):
        #  (log p(w1, w2) / (p(w1)p(w2)) * log f(w1, w2)
        return FeatureRefiner.pmi(jointFreq, indFreqs, allFreq) * log2(jointFreq)

    def createCorrelRefinedFeatTable(self, correls, pValue = 0.05, featNormTable=True):
        keepSet = set() #names of features to keep
        outcomes = set()
        for outcomeField, featCorrels in correls.items():
            outcomes.add(outcomeField)
            for feat, tup in featCorrels.items():
                (r, p, n) = tup[:3]
                if p <= pValue:
                    keepSet.add(feat)

        return self.createNewTableWithGivenFeats(keepSet, '_'.join([x[:3] for x in outcomes]), featNormTable)

    def getCorrelFieldType(self, correlField):
        if correlField == 'state':
            return 'char(2)'
        return None

    def createFeatureTable(self, featureName, featureType = 'VARCHAR(64)', valueType = 'INTEGER', tableName = None, valueFunc = None, correlField=None, extension = None, correl_fieldType = None):
        """Creates a feature table based on self data and feature name"""
        
        #create table name
        if not tableName: 
            valueExtension = ''
            tableName = 'feat$'+featureName+'$'+self.corptable+'$'+self.correl_field
            if valueFunc: 
                tableName += '$' + str(16)+'to'+"%d"%round(valueFunc(16))
            if extension: 
                tableName += '$' + extension

        #find correl_field type:
        where_conditions = """table_schema='%s' AND table_name='%s' AND column_name='%s'"""%(self.corpdb, self.corptable, self.correl_field)
        query = self.qb.create_select_query("information_schema.columns").set_fields(["column_type"]).where(where_conditions)

        correlField = self.getCorrelFieldType(self.correl_field) if not correlField else correlField
        if not correl_fieldType:
            correl_fieldType = query.execute_query()[0][0] if not correlField else correlField
        
        #create sql
        dropTable = self.qb.create_drop_query(tableName)
        # featureType = "VARCHAR(30)" # MAARTEN
        #CREATE TABLE feat_3gram_messages_rand1000_user_id (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, user_id ('bigint(20) unsigned',), 3gram VARCHAR(64), VALUE INTEGER
        #sql = """CREATE TABLE %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id %s, feat %s, value %s, group_norm DOUBLE, feat_norm DOUBLE, KEY `correl_field` (`group_id`), KEY `feature` (`feat`)) CHARACTER SET utf8 COLLATE utf8_general_ci""" %(tableName, correl_fieldType, featureType, valueType)

        #sql = """CREATE TABLE %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id %s, feat %s, value %s, group_norm DOUBLE, KEY `correl_field` (`group_id`), KEY `feature` (`feat`)) CHARACTER SET %s COLLATE %s ENGINE=%s""" %(tableName, correl_fieldType, featureType, valueType, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()], dlac.DEF_MYSQL_ENGINE)

        createTable = self.qb.create_createTable_query(tableName)
        createTable = createTable.add_columns([
            Column("id", "INT", unsigned=True, primary_key=True, nullable=False, auto_increment=True), 
            Column("group_id", correl_fieldType), 
            Column("feat", featureType), 
            Column("value", valueType), 
            Column("group_norm", "DOUBLE")])
        createTable = createTable.add_mul_keys({"correl_field": "group_id", "feature": "feat"})
        createTable = createTable.set_character_set(self.encoding).set_collation(dlac.DEF_COLLATIONS[self.encoding.lower()]).set_engine(dlac.DEF_MYSQL_ENGINE)

        #run sql
        dropTable.execute_query()
        createTable.execute_query()

        return  tableName;


    def createFeatTableByDistinctOutcomes(self, outcomeGetter, controlValuesToAvg = [], outcomeRestriction = None, nameSuffix=None ):
        """Creates a new feature table, by combining values based on an outcome, then applies an averaging based on controls"""
        ##TODO: perform outcome restriction by using group freq thresh instead of uwt, for flexibility
        featureTable = self.featureTable
        outcomeTable = outcomeGetter.outcome_table
        assert len(outcomeGetter.outcome_value_fields) < 2, 'Currently, only allowed to specify one outcome.'
        outcomeField = outcomeGetter.outcome_value_fields[0]
        controlField = None
        if outcomeGetter.outcome_controls: 
            assert len(outcomeGetter.outcome_controls) < 2, 'Currently, only allowed to specify one control.'
            controlField = outcomeGetter.outcome_controls[0]
            if len(controlValuesToAvg) < 1:
                dlac.warn("getting distinct values for controls")
                controlValuesToAvg = outcomeGetter.getDistinctOutcomeValues(outcome = controlField, includeNull = False, where=outcomeRestriction)

        #create new table name:
        nameParts = featureTable.split('$')
        nameParts = [part.replace('16to', '') for part in nameParts]
        nameParts = [part.replace('messages', 'msgs') for part in nameParts]
        newTables = []
        nameSuffix = '' if not nameSuffix else '_%s'%(nameSuffix,)
        if controlField:
            for value in controlValuesToAvg:
                controlGroupName = outcomeField + '_' + controlField + '_' + str(value)
                newTables.append('feat_grpd'+ nameSuffix +'$' + '$'.join(nameParts[1:3]) + '$' + controlGroupName + '$' + '$'.join(nameParts[4:]))
        else: 
            newTables.append('feat_grpd'+ nameSuffix +'$' + '$'.join(nameParts[1:3]) + '$' + outcomeField + '$' + '$'.join(nameParts[4:]))

        #1. create table where outcome is group_id and insert values
        for newTable in newTables:
            drop = """DROP TABLE IF EXISTS %s""" % (newTable)
            sql = "create table %s like %s" % (newTable, featureTable)
            mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            sql = 'ALTER TABLE %s ADD COLUMN `N` int(16) not null default -1'%(newTable)
            mm.execute(self.corpdb, self.dbCursor, sql, mysql_config_file=self.mysql_config_file)
            sql = 'ALTER TABLE %s CHANGE feat_norm std_dev FLOAT' % newTable;
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            mm.standardizeTable(self.corpdb, self.dbCursor, newTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            
        outres = outcomeRestriction
        outres = outres + ' AND ' if outres else '' #only need and if it exists
        if controlField:
            for outcomeValue, cntrlcounts in \
                    outcomeGetter.getDistinctOutcomeAndControlValueCounts(control = controlField, includeNull = False, where=outcomeRestriction).items():
                for cvalue, count in cntrlcounts.items():
                    if cvalue in controlValuesToAvg:
                        newTable = 'feat_grpd'+ nameSuffix + '$' + '$'.join(nameParts[1:3]) + '$' + outcomeField + '_' + controlField + '_' + str(cvalue) + '$' + '$'.join(nameParts[4:])
                        print("on %s %s and %s %s, count: %d" % (outcomeField, str(outcomeValue), controlField, str(cvalue), count))
                        sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) SELECT age, feat, total_freq, mean_rel_freq, SQRT((N_no_zero*(POW((mean_no_zero - mean_rel_freq), 2) + std_no_zero*std_no_zero) + (N - N_no_zero)*(mean_rel_freq * mean_rel_freq)) / N) as std, N  from (SELECT b.%s, feat, SUM(value) as total_freq, SUM(group_norm)/%d as mean_rel_freq, AVG(group_norm) as mean_no_zero, std(group_norm) as std_no_zero, %d as N, count(*) as N_no_zero FROM %s AS a, %s AS b WHERE %s b.%s = '%s' AND b.%s = '%s' AND b.user_id = a.group_id group by b.%s, a.feat) as stats" % (newTable, outcomeField, count, count, featureTable, outcomeTable, outres, controlField, str(cvalue), outcomeField, str(outcomeValue), outcomeField)
#SELECT age, feat, total_freq, mean_rel_freq, SQRT((N_no_zero*(POW((mean_no_zero - mean_rel_freq), 2) + std_no_zero*std_no_zero) + (N - N_no_zero)*(mean_rel_freq * mean_rel_freq)) / N) as std, N  from (
#SELECT b.age, feat, SUM(value) as total_freq, SUM(group_norm)/390 as mean_rel_freq, AVG(group_norm) as mean_no_zero, std(group_norm) as std_no_zero, 390 as N, count(*) as N_no_zero FROM feat$1gram$messages_en$user_id$16to16$0_01 AS a, masterstats_andy AS b WHERE UWT >= 1000 AND b.age = '45' AND b.user_id = a.group_id group by b.age, a.feat) as a             
                        mm.execute(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                    else:
                        print("skipping %s %s and %s %s, count: %d because control value not in list" % (outcomeField, str(outcomeValue), controlField, str(cvalue), count))
        else: #no controls to avg
            # Maarten
            correspondences = outcomeGetter.getGroupAndOutcomeValues()
            correspondences_inv = {}
            for k,v in correspondences:
                correspondences_inv[v] = correspondences_inv.get(v,[])
                correspondences_inv[v].append(k)
            correspondences = correspondences_inv
            total_sum_values = {i[0]: int(i[1]) for i in self.getSumValuesByGroup()}

            i = 0
            j = 0
            for outcomeValue, groups in correspondences.items():
                i += 1
                rows = []
                groups_nonZero = [g for g in groups if g in total_sum_values]
                for feat, values, gns, Nfeats in self.yieldGroupNormsWithZerosByFeat(groups = groups, values = True):
                    if not values: continue

                    sum_value = sum(values.values())
                    total_sum_value = sum(total_sum_values[g] for g in groups_nonZero)
                    group_norm = float(sum_value)/total_sum_value
                    std_dev = std(list(gns.values()))
                    N = len(gns)
                    rows.append([outcomeValue, feat, sum_value, group_norm, std_dev, N])
                    if len(rows) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                        sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) " % newTable
                        sql += "VALUES (%s)" % ', '.join('%s' for r in rows[0]) 
                        mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                        j += len(rows)
                        print("    wrote %d rows [finished %d outcome_values]" % (j, i))
                        rows = []
                    
                if rows:
                    sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) " % newTable
                    sql += "VALUES (%s)" % ', '.join('%s' for r in rows[0])
                    mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                    j += len(rows)  
                    print("    wrote %d rows [finished %d outcome_values]" % (j, i))
                print("Inserted into %s" % newTable)

                        
                """
                for outcomeValue, count in outcomeGetter.getDistinctOutcomeValueCounts(includeNull = False, where=outcomeRestriction).iteritems():
                
                newTable = 'feat_grpd'+ '$' + '$'.join(nameParts[1:3]) + '$' + outcomeField + '$' + '$'.join(nameParts[4:])
                print "on %s %s, count: %d (no control)" % (outcomeField, str(outcomeValue), count)
                sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) SELECT %s, feat, total_freq, mean_rel_freq, SQRT((N_no_zero*(POW((mean_no_zero - mean_rel_freq), 2) + std_no_zero*std_no_zero) + (N - N_no_zero)*(mean_rel_freq * mean_rel_freq)) / N) as std, N  from (SELECT group_id, feat, SUM(value) as total_freq, SUM(group_norm)/%d as mean_rel_freq, AVG(group_norm) as mean_no_zero, std(group_norm) as std_no_zero, count(1) as N_no_zero, %d as N FROM %s) AS a, %s AS b WHERE %s b.%s = '%s' AND b.%s = a.group_id group by b.%s, a.feat" % (newTable, outcomeField,  count, count, featureTable, outcomeTable, outres, outcomeField, str(outcomeValue), self.correl_field ,outcomeField)
                # print "Maarten", self.correl_field, sql
                mm.execute(self.corpdb, self.dbCursor, sql, False)"""
        #2: Combine feature table to take average of controls:
        #controlGroupAvgName = outcomeField + '_' + controlField + 'avg_' + '_'.join(map(lambda v: str(v), controlValuesToAvg))
        if controlField and len(newTables) > 1:  
            controlGroupAvgName = outcomeField + '_' + controlField + 'avg'
            avgTable = 'feat_grpd'+ nameSuffix + '$' + '$'.join(nameParts[1:3]) + '$' + controlGroupAvgName + '$' + '$'.join(nameParts[4:])
            drop = """DROP TABLE IF EXISTS %s""" % (avgTable)
            sql = "create table %s like %s" % (avgTable, newTables[0])
            mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            mm.standardizeTable(self.corpdb, self.dbCursor, avgTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            #create insert fields:
            shortNames = [chr(ord('a')+i) for i in range(len(newTables))]
            tableNames = ', '.join(["%s as %s" % (newTables[i], shortNames[i]) for i in range(len(newTables))])
            values = "(%s)" % (' + '.join(["%s.value" % (name) for name in shortNames])) + ' / ' + str(len(shortNames))
            groupNorms = "(%s)" % (' + '.join(["%s.group_norm" % (name) for name in shortNames])) + ' / ' + str(len(shortNames))
            Ns = ' + '.join(["%s.N" % (name) for name in shortNames])
            stdDev = "(%s)" % (' + '.join(["POW(%s.group_norm - (%s), 2) + POW(%s.std_dev, 2)" % (name, groupNorms, name) for name in shortNames])) + ' / ' + str(len(shortNames))
            stdDev = "SQRT(%s)" % stdDev
            #stdDev = "SQRT(%s)" % (' + '.join(map(lambda name: "%s.N*(POW(%s.group_norm - %s, 2) + POW(%s.std_dev,2))" % (name, name, groupNorms, name), shortNames))) + ' / ' + Ns
            #stdDev = "(%s)" % (' + '.join(map(lambda name: "%s.std_dev_no_zero" % (name), shortNames))) + ' / ' + str(len(shortNames))


            #create joins
            groupIds = ["%s.group_id" % (name) for name in shortNames]
            feats = ["%s.feat" % (name) for name in shortNames]
            groupIdJoins = []
            featJoins = []
            for i in range(len(groupIds) - 1):
                groupIdJoins.append('%s = %s' % (groupIds[i], groupIds[i+1]))
                featJoins.append('%s = %s' % (feats[i], feats[i+1]))
            groupIdJoins = ' AND '.join(groupIdJoins)
            featJoins = ' AND '.join(featJoins)

            #call SQL
            sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) SELECT a.group_id, a.feat, %s, %s, %s, %s FROM %s where %s AND %s" % \
                (avgTable, values, groupNorms, stdDev, Ns, tableNames, groupIdJoins, featJoins)
            print("Populating AVG table with command: %s" % sql)
            mm.execute(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)


    def createAggregateFeatTableByGroup(self, valueFunc = lambda d: d):
        """combines feature tables, and groups by the given group field"""
        
        featureTable = self.featureTable

        (_, name, oldCorpTable, oldGroupField) = featureTable.split('$')[:4]
        theRest = featureTable.split('$')[4:]

        

        newTable = 'feat$agg_'+name[:12]+'$'+oldCorpTable+'$'+self.correl_field # +'$'+'$'.join(theRest)
        drop = """DROP TABLE IF EXISTS %s""" % (newTable)
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        sql = """CREATE TABLE %s like %s""" % (newTable, featureTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        sql = """ALTER TABLE %s MODIFY group_id VARCHAR(255)""" % (newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.standardizeTable(self.corpdb, self.dbCursor, newTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        mm.disableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        
        dlac.warn("Inserting group_id, feat, and values")
        sql = "INSERT INTO %s (group_id, feat, value, group_norm) SELECT m.%s, f.feat, sum(f.value), 0 FROM %s AS f, %s AS m where m.%s = f.group_id GROUP BY m.%s, f.feat" % (newTable,self.correl_field, featureTable, self.corptable, oldGroupField, self.correl_field)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Recalculating group_norms")
        sql = "UPDATE %s a INNER JOIN (SELECT group_id,sum(value) sum FROM %s GROUP BY group_id) b ON a.group_id=b.group_id SET a.group_norm=a.value/b.sum" % (newTable,newTable)
        
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
  
        dlac.warn("Done inserting.\nEnabling keys.")
        mm.enableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        dlac.warn("done.")
        self.featureTable = newTable
        return newTable

    def createTfIdfTable(self, ngram_table):
        '''
        Creates new feature table where group_norm = tf-idf (term frequency-inverse document frequency)
        :param ngram_table: table containing words/ngrams, collocs, etc...

        Written by Phil
        '''

        # tf-idf = tf*idf

        # tf (term frequency) is simply how frequently a term occurs in a document (group_norm for a given group_id)

        # each feat's idf = log(N/dt)
        # N = number of documents in total (i.e. count(distinct(group_id))
        # df (document frequency) = number of documents where feat was used in (i.e. count(distinct(group_id)) where feat = 'feat')

        # create new feature table
        feat_name_grabber = re.compile(r'^feat\$([^\$]+)\$') 
        feat_name = feat_name_grabber.match(ngram_table).group(1) # grabs feat_name (i.e. 1gram, 1to3gram)

        short_name = 'tf_idf_{}'.format(feat_name)
        idf_table = self.createFeatureTable(short_name, valueType = 'DOUBLE')

        #getting N
        sql = "SELECT COUNT(DISTINCT group_id) FROM %s" % ngram_table
        N = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)[0][0]

        feat_counts = self.getFeatureCounts() #tuples of: feat, count (number of groups feature appears with)

        dlac.warn('Inserting idf values into new table')
        counter = 0
        rows = []
        for (feat, dt) in feat_counts:
            idf = log(N/float(dt))

            # get (group_id, group_norm) where feat = feat
            # clean_feat = mm.MySQLdb.escape_string(feat.encode('utf-8')) 

            sql = """SELECT group_id, value, group_norm from %s WHERE feat = \'%s\'"""%(ngram_table, mm.MySQLdb.escape_string(feat.encode('utf-8')).decode('utf-8'))

            group_id_freq = mm.executeGetList(self.corpdb, self.dbCursor, sql, warnQuery=False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

            for (group_id, value, tf) in group_id_freq:
                tf_idf = tf * idf

                rows.append([group_id, mm.MySQLdb.escape_string(feat.encode('utf-8')).decode('utf-8'), value, tf_idf])
                if len(rows) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                    sql = "INSERT INTO %s (group_id, feat, value, group_norm) " % idf_table
                    sql += "VALUES (%s)" % ', '.join('%s' for r in rows[0]) 
                    mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                    rows = []

                if (counter % 50000 == 0):
                    print('%d tf_idf values inserted!' % (counter))
                counter += 1


        if rows:
            sql = "INSERT INTO %s (group_id, feat, value, group_norm) " % idf_table
            sql += "VALUES (%s)" % ', '.join('%s' for r in rows[0]) 
            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn('Finished inserting.')
        return idf_table

    def createSmoothedFeatTable(self, method, mu = None, logRate = False, groupFreqThresh = 0, sqrtRate = False, zeroGap = False):
        """Creates a dense copy of an n-gram table whose group_norm is a smoothed estimate of each feature's rate
        (--smooth_feats; the estimators are in smoothing.py). value keeps the raw count (0 where a group never
        used the feature), so the new table has a row for every feature in every group above groupFreqThresh
        (with zeroGap and no logRate, only the rows of groups that used the feature).

        Parameters
        ----------
        method : str
            one of smoothing.METHODS
        mu : :obj:`float`, optional
            constant prior strength (in words) for dirichlet and logodds_idp; default: estimated for each group
            (for dirichlet with logRate, the corpus-wide estimate)
        logRate : :obj:`boolean`, optional
            store the posterior expected log rate (dirichlet, nb, pln, zinb)
        groupFreqThresh : :obj:`int`, optional
            only groups with at least this many words
        sqrtRate : :obj:`boolean`, optional
            store the square root of the smoothed rate (dirichlet, nb, pln, zinb)
        zeroGap : :obj:`boolean`, optional
            groups that never used a feature keep 0 (with logRate: a value below the feature's smallest log rate)
            rather than a smoothed rate, so used and unused stay apart (dirichlet, nb, pln, zinb); rates and square
            roots then need only the rows of groups that used the feature, as in an unsmoothed table

        Returns
        -------
        str
            the new table (for hurdle, the table of log rates; its 0/1 "used" part is the same name ending in B)
        """
        from . import smoothing
        assert method in smoothing.METHODS, "--smooth_feats method must be one of: %s" % ', '.join(smoothing.METHODS)
        if logRate and method not in smoothing.LOG_METHODS:
            dlac.warn("--smooth_log does not apply to %s (already on a log scale, or has its own log part); ignoring it" % method)
            logRate = False
        if (sqrtRate or zeroGap) and method not in smoothing.LOG_METHODS:
            dlac.warn("--smooth_sqrt and --smooth_zero_gap apply only to %s; ignoring them" % ', '.join(smoothing.LOG_METHODS))
            sqrtRate = zeroGap = False
        if sqrtRate and logRate:
            dlac.warn("--smooth_sqrt and --smooth_log are alternatives; choose one", attention=True)
            sys.exit(1)
        sparseOut = zeroGap and not logRate #unused groups are 0: their rows can be left out, as in the original table
        featureTable = self.featureTable
        pieces = featureTable.split('$')
        featType = pieces[1]
        perGroupMu = method in ('dirichlet', 'logodds_idp') and not mu
        orderField = lambda f: ('BINARY ' + f) if self.db_type == 'mysql' else f #exact order, whatever the collation

        #1. each group's total n-grams (for each n) from the meta tables
        nMatch = re.match(r'^(\d+)(?:to(\d+))?gram', featType)
        ns = list(range(int(nMatch.group(1)), int(nMatch.group(2) or nMatch.group(1)) + 1)) if nMatch else [None]
        featN = (lambda feat: len(feat.split())) if len(ns) > 1 else (lambda feat: ns[0])
        totals = dict() #n => {group_id: total n-grams}
        for n in ns:
            metaTable = '$'.join(['feat', 'meta_' + (featType if len(ns) == 1 else '%dgram' % n)] + pieces[2:4])
            if n is not None and self.data_engine.tableExists(metaTable):
                sql = self.qb.create_select_query(metaTable).set_fields(['group_id', 'value']).where("feat = '_total%dgrams'" % n)
                totals[n] = dict((g, float(v)) for g, v in sql.execute_query())
        inferTotals = len(totals) < len(ns) or not all(totals.values())
        if inferTotals:
            dlac.warn("No meta table with n-gram totals for %s: taking each group's total as value / group_norm, "
                      "which assumes group_norm is the plain relative frequency" % featureTable, attention=True)
            totals = dict((n, dict()) for n in ns)

        #2. one pass over the table: rows, features, and their counts over the groups kept
        wordCounts = self.getGroupWordCounts() if groupFreqThresh else None
        keepGroups = lambda gs: set(g for g in gs if wordCounts.get(g, 0) >= groupFreqThresh) if groupFreqThresh else set(gs)
        groups = None if inferTotals else keepGroups(g for t in totals.values() for g in t)
        dlac.warn("Reading %s" % featureTable)
        numRows, featSums, rowGroups = 0, dict(), set()
        for (gid, feat, value, gn) in self.getFeatAllSS(featNorm=False):
            numRows += 1
            if inferTotals:
                rowGroups.add(gid)
                if gn and gid not in totals[featN(feat)]:
                    totals[featN(feat)][gid] = value / float(gn)
            if groups is None or gid in groups:
                featSums[feat] = featSums.get(feat, 0) + value
        if inferTotals:
            groups = keepGroups(rowGroups)
            featSums = None #recounted below over the groups kept
        groupList = sorted(groups)
        assert len(groupList) > 1, "--smooth_feats needs at least two groups (is group_freq_thresh too high?)"
        groupIdx = dict((g, i) for i, g in enumerate(groupList))
        if featSums is None:
            featSums = dict()
            for (gid, feat, value, gn) in self.getFeatAllSS(featNorm=False):
                if gid in groupIdx: featSums[feat] = featSums.get(feat, 0) + value
        Nvec = dict((n, np.array([totals[n].get(g, 0.0) for g in groupList])) for n in ns)
        Ntot = dict((n, Nvec[n].sum()) for n in ns)
        rowsOut = len(groupList) * len(featSums)
        dlac.warn(" %d groups x %d features = %d rows (the original table has %d)" % (len(groupList), len(featSums), rowsOut, numRows))
        if not sparseOut and rowsOut > dlac.SMOOTH_DENSE_WARN * numRows:
            dlac.warn("The smoothed table is dense and will have %.0fx the rows of %s (%d vs %d). Consider filtering "
                      "features first (--feat_occ_filter)." % (rowsOut / float(numRows), featureTable, rowsOut, numRows), attention=True)

        #3. dirichlet / logodds_idp: each group's mu (leave-one-out likelihood), unless given
        groupMu = mu
        if perGroupMu:
            dlac.warn("Estimating each group's mu (leave-one-out likelihood)")
            pw = dict((feat, s / Ntot[featN(feat)]) for feat, s in featSums.items())
            bestMu = np.full(len(groupList), np.nan)
            totalLL = np.zeros(len(smoothing.MU_GRID))
            def addGroup(gid, byN):
                if gid not in groupIdx or not byN: return
                ll = sum(smoothing.looLogLikGrid(cs, ps, Nvec[n][groupIdx[gid]]) for n, (cs, ps) in byN.items() if Nvec[n][groupIdx[gid]] > 1)
                if np.ndim(ll):
                    bestMu[groupIdx[gid]] = smoothing.MU_GRID[np.argmax(ll)]
                    totalLL[:] += ll
            sql = self.qb.create_select_query(featureTable).set_fields(['group_id', 'feat', 'value']).order_by([(orderField('group_id'), 'ASC')])
            current, byN = None, dict()
            for (gid, feat, value) in self.data_engine.execute_get_SSCursor(sql.toString()):
                if gid != current:
                    addGroup(current, byN)
                    current, byN = gid, dict()
                cs, ps = byN.setdefault(featN(feat), ([], []))
                cs.append(value)
                ps.append(pw[feat])
            addGroup(current, byN)
            globalMu = smoothing.MU_GRID[np.argmax(totalLL)]
            bestMu = np.where(np.isnan(bestMu), globalMu, bestMu)
            groupMu = smoothing.shrinkGroupMus(bestMu, globalMu, Nvec[ns[0]], dlac.DEF_SMOOTH_MIN_WORDS, dlac.DEF_SMOOTH_MU_RANGE)
            dlac.warn(" corpus-wide mu: %.4g; per-group mu: median %.4g (%.4g to %.4g)" % (globalMu, np.median(groupMu), groupMu.min(), groupMu.max()))
            if method == 'dirichlet' and logRate:
                #in log, a never-used word's value is driven by its group's mu, a factor every feature would share
                #(and that tracks the group's language): log rates use the one corpus-wide mu
                dlac.warn(" log rates use the corpus-wide mu (%.4g) for every group" % globalMu)
                groupMu = globalMu

        #4. the new table(s)
        muLabel = ('%g' % mu).replace('.', '_').replace('+', '') if mu else ''
        suffix = {'dirichlet': 'dir' + muLabel, 'nb': 'nb', 'pln': 'pln', 'zinb': 'zinb', 'logodds_idp': 'loi' + muLabel,
                  'logodds': 'lo', 'hurdle': 'hurL'}[method] + ('L' if logRate else 'S' if sqrtRate else '')
        suffix = ('zg' if zeroGap else '') + suffix
        newTables = [featureTable + '$' + suffix] + ([featureTable + '$hurB'] if method == 'hurdle' else [])
        for newTable in newTables:
            if self.db_type == 'mysql' and len(newTable) > 64:
                dlac.warn("%s is longer than MySQL's 64 characters; use a shorter message table name" % newTable, attention=True)
                sys.exit(1)
            self.qb.create_drop_query(newTable).execute_query()
            self.qb.create_createTable_query(newTable).like(featureTable).execute_query()
            self.data_engine.standardizeTable(newTable, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)
            self.data_engine.disable_table_keys(newTable)
        wsqls = [self.qb.create_insert_query(t).set_values([("group_id", ""), ("feat", ""), ("value", ""), ("group_norm", "")]) for t in newTables]

        #5. one pass in feature order: smooth each feature over all groups and write it
        dlac.warn("Smoothing (%s%s%s) and writing %s" % (method, ', log rate' if logRate else ', square root' if sqrtRate else '',
                                                         ', zero gap' if zeroGap else '', ', '.join(newTables)))
        toWrite = [[] for t in newTables]
        numFeats, nonFinite = 0, 0
        def smoothFeat(feat, rows):
            c = np.zeros(len(groupList))
            for gid, value in rows:
                if gid in groupIdx: c[groupIdx[gid]] = value
            n = featN(feat)
            N, C, Cw = Nvec[n], Ntot[n], c.sum()
            if not Cw: return [] #only used by groups below group_freq_thresh
            p = Cw / C
            with np.errstate(divide='ignore', invalid='ignore'):
                if method == 'dirichlet': values = [smoothing.dirichlet(c, N, p, groupMu, logRate)]
                elif method == 'nb': values = [smoothing.nb(c, N, p, smoothing.nbDispersion(c, N, p), logRate)]
                elif method == 'pln': values = [smoothing.pln(c, N, p, smoothing.nbDispersion(c, N, p), logRate)]
                elif method == 'zinb': values = [smoothing.zinb(c, N, *smoothing.zinbFit(c, N), log=logRate, floor=0.5 / C)]
                elif method == 'logodds_idp': values = [smoothing.logoddsIDP(c, N, Cw, C, p, groupMu)]
                elif method == 'logodds': values = [smoothing.logodds(c, N, Cw, C)]
                else: values = list(reversed(smoothing.hurdle(c, N, p, smoothing.nbDispersion(c, N, p)))) #(log rate, used)
                if sqrtRate: values = [np.sqrt(v) for v in values]
                if zeroGap: values = [smoothing.zeroGap(v, c, logRate) for v in values]
            return values, c
        def writeFeat(feat, rows):
            nonlocal numFeats, nonFinite
            result = smoothFeat(feat, rows)
            if not result: return
            values, c = result
            numFeats += 1
            keep = np.nonzero(c)[0] if sparseOut else np.arange(len(groupList))
            for t, vals in enumerate(values):
                nonFinite += int((~np.isfinite(vals)).sum())
                toWrite[t].extend(zip([groupList[i] for i in keep], [feat] * len(keep), c[keep].astype(int).tolist(), vals[keep].tolist()))
                if len(toWrite[t]) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                    wsqls[t].execute_query(toWrite[t])
                    toWrite[t] = []
            if numFeats % 5000 == 0: dlac.warn("  %d features smoothed" % numFeats)
        sql = self.qb.create_select_query(featureTable).set_fields(['group_id', 'feat', 'value']).order_by([(orderField('feat'), 'ASC')])
        current, rows = None, []
        for (gid, feat, value) in self.data_engine.execute_get_SSCursor(sql.toString()):
            if feat != current:
                if rows: writeFeat(current, rows)
                current, rows = feat, []
            rows.append((gid, value))
        if rows: writeFeat(current, rows)
        for t, wsql in enumerate(wsqls):
            if toWrite[t]: wsql.execute_query(toWrite[t])
        if nonFinite:
            dlac.warn("%d smoothed values were not finite (nan or inf)" % nonFinite, attention=True)

        dlac.warn("Done: %d features. Enabling keys." % numFeats)
        for newTable in newTables:
            self.data_engine.enable_table_keys(newTable)
        self.featureTable = newTables[0]
        return newTables[0]

    def _calc_pmi_iter(self, multigram_counts_iter, onegram_counts_iter, total_count, tokenize_func=lambda x:x.split()):
        '''
        :param multigram_counts_iter: iterator yields dicts, dicts have columns "feat" and "count"
            this is the list of multigrams for which we are calculating pmi values
        :param onegram_counts_iter: iterator yields dicts, dicts have columns "feat" and "count"
            this should include all onegrams that are included in the multigrams from multigram_counts_iter
        :param total_count: the total count of words in the body of text from which onegram and multigrams were generated
        :param tokenize_func: a function that take a string and output's a list of words or tokens
        :return: an iterator of multigrams with their associated PMI information
        '''

        onegram_counts_dict = {}
        print("Reading onegrams into a dict...")
        count = 0
        for onegram_count_row in onegram_counts_iter:
            count += 1
            if count % 100000 == 0:
                print("Processing onegram row {}".format(count))
            onegram = onegram_count_row[0]
            onegram_counts_dict[onegram] = onegram_count_row[1]

        for multigram_count_row in multigram_counts_iter:
            multigram_count_row = dict({
                "id": multigram_count_row[0],
                "feat": multigram_count_row[1], 
                "count": multigram_count_row[2]})
            multigram = multigram_count_row["feat"]
            onegrams = tokenize_func(multigram)
            num_tokens = len(onegrams)

            onegram_presence = [onegram in onegram_counts_dict for onegram in onegrams]
            if num_tokens <= 1 or not all(onegram_presence):
                (pmi_val, npmi_val, pmi_filter_val, npmi_filter_val) = (None, None, None, None)

            else:
                count_multigram = multigram_count_row["count"]
                onegram_counts = [onegram_counts_dict[onegram] for onegram in onegrams]
                pmi_val = self._colloc_pmi(count_multigram, onegram_counts, total_count)
                npmi_val = self._colloc_pmi(multigram_count_row["count"], onegram_counts, total_count, normalize=True)
                pmi_filter_val = pmi_val/(num_tokens - 1)
                npmi_filter_val = npmi_val/(num_tokens - 1)

            multigram_count_row.update({'pmi':pmi_val, 'pmi_filter_val':pmi_filter_val, 'npmi':npmi_val, 'npmi_filter_val':npmi_filter_val})
            yield multigram_count_row

    def _colloc_pmi(self, count_colloc, counts_onegrams, total_count, normalize = False, useAndyDenom = False):
        '''
        Source: https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf
        :param prob_colloc: float from 0 to 1
        :param prob_ngrams: list of floats from 0 to 1
        :return: float from ??? to ???
        '''
        prob_colloc = float(count_colloc)/float(total_count)
        probs_onegrams = [float(onegram_count)/float(total_count) for onegram_count in counts_onegrams]
        if useAndyDenom:
            raise NotImplementedError
        else:
            denom = float(reduce(mul, probs_onegrams, 1))

        ratio = prob_colloc/denom
        pmi_val = math.log(ratio, math.e)
        if normalize:
            normalization_factor = -1 * math.log(prob_colloc)
            npmi_val = pmi_val/normalization_factor
            return npmi_val
        else:
            return pmi_val

    def _colloc_lnpmi_vals(self, colloc_list, counts, total_word_count):
        '''
        :param colloc_list - list of collocs that we want values calculated for
        :param counts - dict keyed by colloc, where multiple tokens are separated by spaces
        :param total_word_count - int
        :returns dict iter
        '''
        for colloc in colloc_list:
            ##TODO - find way to deal with missing onegrams or two grams
            onegrams = colloc.split()
            num_tokens = len(onegrams)
            prob_colloc = float(counts[colloc])/float(total_word_count)
            normalization_factor = -1 * math.log(prob_colloc)
            onegram_probs = [float(counts[onegram])/float(total_word_count) for onegram in onegrams]
            pmi_denom = reduce(mul, onegram_probs, 1)
            if num_tokens >= 3:
                try:
                    count_prod1 = counts[onegrams[0]] * counts[" ".join(onegrams[1:])]
                    count_prod2 = counts[" ".join(onegrams[:-1])] * counts[onegrams[-1]]
                except Exception as e:
                    raise e
                andy_denom = float(max(count_prod1, count_prod2))/math.pow(total_word_count, 2)
            pmi_val = math.log(prob_colloc/pmi_denom, math.e)
            lpmi_val = math.log(prob_colloc/andy_denom, math.e) if num_tokens >= 3 else pmi_val
            npmi_val = pmi_val/normalization_factor
            lnpmi_val = lpmi_val/normalization_factor
            yield {'feat':colloc, 'num_tokens':num_tokens, 'pmi':pmi_val, 'npmi':npmi_val, 'lpmi':lpmi_val, 'lnpmi':lnpmi_val}


    def creatCollocScores(self, ufeat_table):
        """"""
        if not isinstance(ufeat_table, str):
            ufeat_table = self.corptable
        db_eng = mif.get_db_engine(self.corpdb, mysql_config_file=self.mysql_config_file)

        ufeat_multigram_table = "ufeat$" + ufeat_table

        dropQuery = self.qb.create_drop_query(ufeat_multigram_table)
        dropQuery.execute_query()

        columns = [
            Column("id", "BIGINT", primary_key=True, auto_increment=True),
            Column("feat", "VARCHAR(102)"),
            Column("count", "BIGINT")]
        createQuery = self.qb.create_createTable_query(ufeat_multigram_table).add_columns(columns).add_mul_keys({"feat": "feat"})
        createQuery = createQuery.set_character_set(self.encoding).set_collation(dlac.DEF_COLLATIONS[self.encoding.lower()])
        createQuery = createQuery.set_engine(dlac.DEF_MYSQL_ENGINE)
        createQuery.execute_query()
        
        selectQuery = self.qb.create_select_query(self.featureTable).set_fields(["feat", "SUM(value) count"]).group_by(["feat"])
        insertQuery = self.qb.create_insert_query(ufeat_multigram_table).set_values([("feat", ''), ("count", '')]).values_from_select(selectQuery)
        insertQuery.execute_query()

        print("Extending table if necessary...")
        new_cols = OrderedDict()
        new_cols['pmi'] = 'DOUBLE';
        new_cols['pmi_filter_val'] = 'DOUBLE';
        new_cols['npmi'] = 'DOUBLE';
        new_cols['npmi_filter_val'] = 'DOUBLE';
        group_column = self.featureTable.split('$')[3]
        pocc_column = "pocc_{}_gft0".format(group_column)
        new_cols[pocc_column] = 'DOUBLE';
        mif.extend_table(self.data_engine, ufeat_multigram_table, new_cols)

        print("Querying input data...")
        selectQuery = self.qb.create_select_query(self.wordTable).set_fields(["SUM(value)"])
        total_count = selectQuery.execute_query()[0][0]

        ###AHHHH this MUST be grouped!!!!
        selectQuery = self.qb.create_select_query(self.wordTable).set_fields(["feat", "SUM(value) count"]).group_by(["feat"])
        onegram_counts_iter = selectQuery.execute_query()
        where_condition = "pmi IS NULL AND feat LIKE '%% %%' AND count > 1"
        selectQuery = self.qb.create_select_query(ufeat_multigram_table).set_fields(["id", "feat", "count"]).where(where_condition)
        multigram_counts_iter = selectQuery.execute_query()

        pmi_iter = self._calc_pmi_iter(multigram_counts_iter, onegram_counts_iter, total_count)

        print("Processing npmi data...")
        mif.mysql_update(self.data_engine, ufeat_multigram_table, pmi_iter)

        print("Done npmi.")

        ### annotate pocc
        selectQuery = self.qb.create_select_query(self.featureTable).set_fields(["count(distinct group_id)"])
        num_groups_tot = selectQuery.execute_query()[0][0]

        print("Loading group counts by feat...")
        #group_count_sql = "SELECT feat, count(*) group_count FROM {} GROUP BY feat".format(self.featureTable)
        #df_group_counts = pd.read_sql(group_count_sql, db_eng, index_col="feat")
        selectQuery = self.qb.create_select_query(self.featureTable).set_fields(["feat", "count(*) group_count"]).group_by(["feat"])
        df_group_counts = {feat: count for feat, count in selectQuery.execute_query()}

        print("Loading ufeat data...")
        selectQuery = self.qb.create_select_query(ufeat_multigram_table).set_fields(["id", "feat"]).where("count > 1")
        feat_iter = selectQuery.execute_query()

        def pocc_gen():
            count = 0
            for (id, feat) in feat_iter:
                num_groups = df_group_counts[feat]
                yield {'id':id, pocc_column:float(num_groups)/num_groups_tot}
        pocc_dict_iter = pocc_gen()

        print("Updating ufeat data with pocc values...")
        mif.mysql_update(self.data_engine, ufeat_multigram_table, pocc_dict_iter, log_every=10000)
        return ufeat_multigram_table


    def createArchetypeFeats(self, archetypeFeats, meanCenter=True, standardize=False, featNormTable=False, \
                             metric = lambda v, X: cosine_similarity(v[None,:], X)):
                             #v.dot(X.T) / (norm(X.T,axis=0)*norm(v))): 
                             #np.dot(X, v)/(norm(X, axis=1)*norm(v))):
        """Creates a feature table given a 1gram feature table name, a lexicon table / database name

        Parameters
        ----------
        archetype_feats: str
            Name of base lexicon table

        Returns
        -------
        tableName : str
            Name of created feature table: feat%archetype_lexTable%corptable$archetype_names

        """

        #prepare:
        #from archetypes import ArchetypeCollection, ArchetypeQuantifier 
        #main similarity metric: ArchetypeQuantifier.static_analyzer
        #TODO: update to note require reading column names from table
        
        [_, archFeatName, archCorpus, archCategory] = archetypeFeats.split('$')[:4]
        archGetter = FeatureGetter(self.db_type, self.corpdb, archCorpus, archCategory, self.mysql_config_file, self.message_field, self.messageid_field, self.encoding, self.use_unicode, self.lexicondb, archetypeFeats, wordTable = self.wordTable)
        
        featureTable = self.featureTable
        dlac.warn("adding archetype " + archetypeFeats + " features for " + featureTable)
        # probably not needed: wordGetter = self.getWordGetter()

        ##STEP 1: get the archetype vectors (group_id is category)
        (archGNs, archFeatureNames) = archGetter.getGroupNormsWithZeros()
        archFeatureNames = list(archFeatureNames)
        dlac.warn("\narchetypes: %s; number of features: %d" % (str(archGNs.keys()), len(archFeatureNames)))
    
        ##STEP 2: Get document features
        #(groupNorms, featureNames) = self.getGroupNormsSparseFeatsFirst()
        (gns, featureNames) = self.getGroupNormsWithZeros()
        groups = list(gns.keys())
        X = np.array([[gns[gid][feat] for feat in archFeatureNames] for gid in gns.keys()])
        #assert all([isinstance(X,numbers.Number) for x in X]), "Data is corrupt, there are non float elements in the group norms (some might be NULL?)"
        Xmean = X.mean(axis=0)
        if meanCenter:
            X = X - Xmean
        if standardize:
            X = X/X.std(axis=0,keepdims=True)
        dlac.warn("Data being compared to archetypes (groups, feats): %s" % str(X.shape))

        #STEP 3 Iterate over archs to calculate similarity
        sims = {} #holds the architype similarities
        for archName, archGNs in archGNs.items():
            dlac.warn("calculating for: "+archName)
            archVector = np.array([archGNs[feat] for feat in archFeatureNames])
            if meanCenter:
                #archVector = archVector - archVector.mean()
                archVector = archVector - Xmean
            if standardize:
                archVector = archVector / archVector.std()
            sims[archName] = metric(archVector, X).flatten().tolist()#[0]
        dlac.warn({k: len(v) for k, v in sims.items()})
        
        ##STEP 4: write the new feature table
        nameLength = max([len(k) for k in sims.keys()])
        featTypeName = "arc_"+archFeatName[:5]+'_'+archCategory[:4]
        tableName = self.createFeatureTable(featTypeName, "VARCHAR(%d)"%nameLength, 'DOUBLE')
        #INSERT:
        #TODO: make this a function!
        for archName, sscores in sims.items():
            wsql = """INSERT INTO """+tableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            rows = [(str(groups[i]), archName, int(np.round(sscores[i])), sscores[i]) for i in range(len(sscores))]
            #pprint([rows[i] for i in np.random.choice(range(len(rows)), 10, False).tolist()])
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, mysql_config_file=self.mysql_config_file)
        dlac.warn("Done Inserting; new table: %s" % tableName)

        return tableName

