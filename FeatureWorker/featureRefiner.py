import re
from collections import OrderedDict
from pprint import pprint

#math / stats:
from numpy import sqrt, array, std, mean, log2, log

#local / nlp
from lib.happierfuntokenizing import Tokenizer #Potts tokenizer

#infrastructure
from featureGetter import FeatureGetter
import fwConstants as fwc
from mysqlMethods import mysqlMethods as mm
from mysqlMethods.mysql_iter_funcs import get_db_engine

class FeatureRefiner(FeatureGetter):
    """Deals with the refinement of feature information already in a table (outputs to new table)"""

    def makeTopicLabelMap(self, topiclexicon, numtopicwords=5, is_weighted_lexicon=False):
        featlabel_tablename = 'feat_to_label$%s$%d'%(topiclexicon, numtopicwords)

        pldb = self.lexicondb
        (plconn, plcur, plcurD) = mm.dbConnect(pldb, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'DROP TABLE IF EXISTS `%s`'%featlabel_tablename
        mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'CREATE TABLE `%s` (`id` int(16) unsigned NOT NULL AUTO_INCREMENT, `term` varchar(128) DEFAULT NULL, `category` varchar(64) DEFAULT NULL, PRIMARY KEY (`id`), KEY `term` (`term`), KEY `category` (`category`) )'%featlabel_tablename
        mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode)

        sql = 'SELECT DISTINCT category FROM %s'%topiclexicon
        categories = map(lambda x: x[0], mm.executeGetList(pldb, plcur, sql))
        label_list = []
        for category in categories:
            if is_weighted_lexicon:
                sql = 'SELECT term, weight from %s WHERE category = \'%s\''%(topiclexicon, category)
                rows = mm.executeGetList(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode)
                top_n_rows = sorted(rows, key=lambda x:x[1], reverse=True)
                terms = map(lambda x: x[0], top_n_rows)
                label = ' '.join(map(str, terms[0:numtopicwords]))
                escaped_label = MySQLdb.escape_string(label)
                sql = 'INSERT INTO `%s` (`term`, `category`) VALUES(\'%s\', \'%s\')'%(featlabel_tablename, category, escaped_label )
                mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode)
            else:
                sql = 'SELECT term from %s WHERE category = \'%s\''%(topiclexicon, category)
                terms = map(lambda x: x[0], mm.executeGetList(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode))
                label = ' '.join(map(str, terms[0:numtopicwords]))
                escaped_label = MySQLdb.escape_string(label)
                sql = 'INSERT INTO `%s` (`term`, `category`) VALUES(\'%s\', \'%s\')'%(featlabel_tablename, category, escaped_label )
                mm.execute(pldb, plcur, sql, charset=self.encoding, use_unicode=self.use_unicode)

        return featlabel_tablename

    def createCombinedFeatureTable(self, featureName = None, featureTables = [], tableName = None):
        """Create a new feature table by combining others"""

        #get best feat column type:
        intGrabber = re.compile(r'\d+')
        longestInt = 12
        longestType = "VARCHAR(12)"
        valueType = 'INTEGER'
        for table in featureTables:
            columns = mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, table)
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
            for func in fwc.POSSIBLE_VALUE_FUNCS:
                if float(func(16)) == float(toNum):
                    valueFunc = func
                    break     
        pocc = None
        poccmatch = re.search(r'\$([_0-9]+)\s*$', featureTables[0])
        if poccmatch:
            pocc = poccmatch.group(1)
        
        #CREATE TABLE:
        if not featureName:
            featNameGrabber = re.compile(r'^feat\$([^\$]+)\$')
            names = []
            for table in featureTables:
                names.append(featNameGrabber.match(table).group(1))
            featureName = '_'.join(names)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%longestInt, valueType, tableName, valueFunc, extension=pocc)
        # Maarten: todo: test if too long and don't disable keys
        mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        for fTable in featureTables:
            mm.execute(self.corpdb, self.dbCursor, "INSERT INTO %s (group_id, feat, value, group_norm) SELECT group_id, feat, value, group_norm from %s;" % (featureTableName, fTable), charset=self.encoding, use_unicode=self.use_unicode)
        
        mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        return featureTableName


    def createTableWithBinnedFeats(self, num_bins, group_id_range, valueFunc = lambda x:x, 
                                   gender=None, genderattack=False, reporting_percent=0.04, outcomeTable = fwc.DEF_OUTCOME_TABLE, skip_binning=False):
        featureTable = self.featureTable
        group_id_range = map(int, group_id_range)
        newTable = featureTable+'$'+str(num_bins)+'b_'+'_'.join(map(str,group_id_range))
        if skip_binning: return newTable

        sql = 'DROP TABLE IF EXISTS %s'%newTable
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = "CREATE TABLE %s like %s" % (newTable, featureTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

        groupNs = mm.executeGetList(self.corpdb, self.dbCursor, 'SELECT group_id, N FROM %s GROUP BY group_id'%self.featureTable, charset=self.encoding, use_unicode=self.use_unicode)
        groupIdToN = dict(groupNs)
        total_freq = sum(map(lambda x:x[1], groupNs))
        bin_size = float(total_freq) / float(num_bins+2)

        num_groups = len(groupNs)
        reporting_int = fwc._getReportingInt(reporting_percent, num_groups)

        # figure out the bins, i.e. if group_id's 1,2,3 total value is greater than "bin_size" our first bin is 1_3.
        fwc.warn('determining the number of bins...')
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
            fwc._report('group_id\'s', gg, reporting_int, num_groups)
        if current_sum >= 0:
            bin_groups[(current_lower_group, current_upper_group)]  = '_'.join(map(str,[current_lower_group, current_upper_group]))

        max_label_length = max(map(len, bin_groups.values()))

        sql = 'ALTER TABLE %s MODIFY COLUMN group_id VARCHAR(%d)'%(newTable, max_label_length) #this action preserves the index
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'ALTER TABLE %s ADD COLUMN `bin_center` float(6) not null default -1.0'%(newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'ALTER TABLE %s ADD COLUMN `bin_center_w` float(6) not null default -1.0'%(newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'ALTER TABLE %s ADD COLUMN `bin_width` int(10) not null default -1'%(newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        mm.disableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode)

        # for each newly denoted bin: e.g. 1_3, 4_5, 6_6, ... get the new feature value counts / group norms; insert them into the new table
        # e.g. 1 'hi' 5, 2 'hi' 10, 3 'hi' 30 ==> 1_3 'hi' 45  (of course include group_norm also)
        fwc.warn('aggreagating the newly binned feature values / group_norms into the new table...')
        isql = 'INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N, bin_center, bin_center_w, bin_width) VALUES (%s)'%(newTable, '%s, %s, %s, %s, %s, %s, %s, %s, %s')
        #isql = 'INSERT INTO %s (group_id, feat, value, group_norm, N, bin_center, bin_width) VALUES (%s)'%(newTable, '%s, %s, %s, %s, %s, %s, %s')
        ii_bins = 0
        num_bins = len(bin_groups.keys())
        reporting_int = fwc._getReportingInt(reporting_percent, num_bins)
        #_warn('#############BIN NUMBER############### [[%d]] #############'%len(bin_groups))
        for (lower_group, upper_group), label in bin_groups.iteritems():
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
            groupFeatValueNorm = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
            #pprint(groupFeatValueNorm)

            totalFeatCountForThisBin = float(0)
            featToValue = {}
            featToSummedNorm = {}
            for group_id, feat, value, norm, sd in groupFeatValueNorm:
            # for group_id, feat, value, norm, N in groupFeatValueNorm:
                if fwc.LOWERCASE_ONLY: feat = str(feat).lower()
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
                               bin_N_sum, bin_center, bin_center_w, bin_width) for k, v in featToValue.iteritems() ]
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, current_batch, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            # print 'N bin sum:', bin_N_sum
            # isql = 'INSERT INTO %s (group_id, feat, value, group_norm, N, bin_center, bin_center_w, bin_width) VALUES (%s)'%(newTable, '%s, %s, %s, %s, %s, %s, %s, %s')
            ii_bins += 1
            fwc._report('group_id bins', ii_bins, reporting_int, num_bins)

        mm.enableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode)
        fwc.warn('Done creating new group_id-binned feature table.')

        outputdata = mm.executeGetList(self.corpdb, self.dbCursor, 'select group_id, N from `%s` group by group_id'%(newTable,), charset=self.encoding, use_unicode=self.use_unicode)
        pprint(outputdata)

        # mm.execute(self.corpdb, self.dbCursor, 'drop table if exists `%s`'%(newTable,))
        return newTable

    def createTableWithRemovedFeats(self, p, minimumFeatSum = 0, groupFreqThresh = 0, setGFTWarning = False):
        """creates a new table with features that appear in more than p*|correl_field| rows, only considering groups above groupfreqthresh"""
        if not setGFTWarning:
            fwc.warn("""group_freq_thresh is set to %s. Be aware that groups might be removed during this process and your group norms will reflect this.""" % (groupFreqThresh), attention=True)
        toKeep = self._getKeepSet(p, minimumFeatSum, groupFreqThresh)
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
        mm.execute(self.corpdb, self.dbCursor, "DROP TABLE IF EXISTS %s" % newTable, charset=self.encoding, use_unicode=self.use_unicode)
        fwc.warn(" %s <new table %s will have %d distinct features.>" %(featureTable, newTable, numToKeep))
        sql = """CREATE TABLE %s like %s""" % (newTable, featureTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        mm.disableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode)
  
        num_at_time = 2000
        total = 0
        toWrite = []
        
        wsql = """INSERT INTO """+newTable+""" (group_id, feat, value, group_norm, feat_norm) values (%s, %s, %s, %s, %s)""" if featNorm else """INSERT INTO """+newTable+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""

        #iterate through each row, deciding whetehr to keep or not
        for featRow in self.getFeatAllSS(featNorm=featNorm):
            #print "%d %d" % (len(featRow), len(toWrite))
            if self.use_unicode and unicode(featRow[1]).lower() in toKeep:
                toWrite.append(featRow)
            elif not self.use_unicode and featRow[1].lower() in toKeep:
                toWrite.append(featRow)
            if len(toWrite) > num_at_time:
            #write those past the filter to the table
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                total+= num_at_time
                if total % 100000 == 0: fwc.warn("%.1fm feature instances written" % (total/float(1000000)))
                toWrite = []

        #catch rest:
        if len(toWrite) > 0:
            #write those past the filter to the table
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        fwc.warn("Done inserting.\nEnabling keys.")
        mm.enableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode)
        fwc.warn("done.")

        self.featureTable = newTable
        return newTable

    def _getKeepSet(self, p, minimumFeatSum = 0, groupFreqThresh = 0):
        """creates a set of features occuring in less than p*|correl_field| rows"""
        #acquire the number of groups (need to base on corp table):
        featureTable = self.featureTable
        totalGroups = self.countGroups(groupFreqThresh)
        assert totalGroups > 0, 'NO GROUPS TO FILTER BASED ON (LIKELY group_freq_thresh IS TOO HIGH)'
        assert p <= 1, 'p_occ > 1 not implemented yet'
        threshold = int(round(p*totalGroups))
        fwc.warn (" %s [threshold: %d]" %(featureTable, threshold))

        #acquire counts per feature (each row will come from a different correl_field)

        featCounts = self.getFeatureCounts(groupFreqThresh) #tuples of: feat, count (number of groups feature appears with)
        
        #apply filter:
        toKeep = set()
        i = 0
        for (feat, count) in featCounts:
            if count >= threshold:
                if self.use_unicode:
                    toKeep.add(unicode(feat).lower())
                else:
                    toKeep.add(feat.lower())
            i += 1
            
            if (i % 1000000) == 0: print "    checked %d features" % i
        
        #apply secondary filter
        if minimumFeatSum > 1:
            featSums = self.getFeatureValueSums()
            for (feat, fsum) in featSums:
                if self.use_unicode:
                    feat = unicode(feat).lower()
                else:
                    feat = feat.lower()
                if feat in toKeep:
                    if fsum < minimumFeatSum:
                        toKeep.remove(feat)
            
        return toKeep

    def addFeatNorms(self, ReCompute = False):
        """Adds the mean normalization by feature (z-score) for each feature"""
        where = None
        if not ReCompute: where = 'feat_norm is null'
        groupNorms = self.getGroupNorms(where = where) #contains group_id, feat, group_norm
        
        fMeans = self.addFeatTableMeans(groupNorms = groupNorms) #mean, std, zero

        wsql = """UPDATE """+self.featureTable+""" SET feat_norm = %s where group_id = %s AND feat = %s"""
        featNorms = []
        num_at_time = 2000
        numWritten = 0
        for (group_id, feat, group_norm) in groupNorms:
            if fwc.LOWERCASE_ONLY: feat = feat.lower()
            if (feat):
                fn = ( ((group_norm - fMeans[feat][0]) / float(fMeans[feat][1]), group_id, feat) )
                featNorms.append(fn)
                if len(featNorms) >= num_at_time:
                    mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, featNorms, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                    featNorms = []
                    numWritten += num_at_time
                    if numWritten % 100000 == 0: fwc.warn("%.1fm feature instances updated out of %dm" % 
                                                        ((numWritten/float(1000000)), len(groupNorms)/1000000))
                                    
        
        #write values back in 
        if featNorms: 
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, featNorms, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        return True

        
    def findMeans(self, field='group_norm', addZeros = True, groupNorms = None):
        """Finds feature means from group norms"""
        if not groupNorms:
            groupNorms = self.getGroupNorms() #contains group_id, feat, group_norm

        #turn into dict of lists:
        fNumDict = dict() 
        for (gid, feat, gn) in groupNorms:
            if fwc.LOWERCASE_ONLY: feat = feat.lower()
            if feat: 
                if not feat in fNumDict:
                    fNumDict[feat] = [gn]
                else:
                    fNumDict[feat].append(gn)

        #add zeros for groups missing featrues
        if addZeros:
            totalGroups = self.countGroups()
            for feat, nums in fNumDict.iteritems():
                difference = totalGroups - len(nums)
                nums.extend([0]*difference)
            
        #calculate values:
        fMeans = dict()
        for feat, nums in fNumDict.iteritems():
            nums = array(map(lambda d: float(d), nums))
            (m, s) = (mean(nums), std(nums))
            z = (0 - m) / float(s)
            fMeans[feat] = (m, s, z)

        return fMeans


    def addFeatTableMeans(self, field='group_norm', groupNorms = None):
        """Add to the feature mean table: mean, standard deviation, and zero_mean for the current feature table"""

        #CREATE TABLE
        meanTable = 'mean$'+self.featureTable
        mm.execute(self.corpdb, self.dbCursor, "DROP TABLE IF EXISTS %s" % meanTable, charset=self.encoding, use_unicode=self.use_unicode)
        featType = mm.executeGetList(self.corpdb, self.dbCursor, "SHOW COLUMNS FROM %s like 'feat'" % self.featureTable)[0][1]
        sql = """CREATE TABLE %s (feat %s, mean DOUBLE, std DOUBLE, zero_feat_norm DOUBLE, PRIMARY KEY (`feat`))""" % (meanTable, featType)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

        fMeans = self.findMeans(field, True, groupNorms)
        fMeansList = [(k, v[0], v[1], v[2]) for k, v in fMeans.iteritems()]
        #print fMeansList #debug

        #WRITE TO TABLE:
        sql = """INSERT INTO """+meanTable+""" (feat, mean, std, zero_feat_norm) VALUES (%s, %s, %s, %s)"""
        mm.executeWriteMany(self.corpdb, self.dbCursor, sql, fMeansList, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        return fMeans


    def createCollocRefinedFeatTable(self, threshold = 3.0, featNormTable=False):
        #n = the number of words in the ngrams
        #uses pmi to remove uncommon collocations:
        featureTable = self.featureTable
        fwc.warn(featureTable)
        wordGetter = self.getWordGetter()
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        jointFreqs = self.getSumValuesByFeat()
        wordFreqs = dict(wordGetter.getSumValuesByFeat())
        allFreqs = wordGetter.getSumValue()

        keepers = set()
        for (colloc, freq) in jointFreqs:
            # words = tokenizer.tokenize(colloc)
            # If words got truncated in the creation of 1grams, we need to account for that
            words = [word[:fwc.VARCHAR_WORD_LENGTH] for word in tokenizer.tokenize(colloc)]
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
        fwc.warn(featureTable)
        wordGetter = self.getWordGetter()
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        jointFreqs = self.getSumValuesByFeat()
        wordFreqs = dict(wordGetter.getSumValuesByFeat())
        allFreqs = wordGetter.getSumValue()

        keepers = set()
        collocPMIs = {}
        count = 0
        print "len(jointFreqs): " + str(len(jointFreqs))
        for (colloc, freq) in jointFreqs:
            count +=1
            if count % 50000 == 0:
                print "calculating pmi for {}th feature".format(count)
            words = [word[:fwc.VARCHAR_WORD_LENGTH] for word in tokenizer.tokenize(colloc)]
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
            print jointFreq, indFreqs, allFreq, words
        return out

    @staticmethod
    def salience(jointFreq, indFreqs, allFreq):
        #  (log p(w1, w2) / (p(w1)p(w2)) * log f(w1, w2)
        return FeatureRefiner.pmi(jointFreq, indFreqs, allFreq) * log2(jointFreq)

    def createCorrelRefinedFeatTable(self, correls, pValue = 0.05, featNormTable=True):
        keepSet = set() #names of features to keep
        outcomes = set()
        for outcomeField, featCorrels in correls.iteritems():
            outcomes.add(outcomeField)
            for feat, tup in featCorrels.iteritems():
                (r, p, n) = tup[:3]
                if p <= pValue:
                    keepSet.add(feat)

        return self.createNewTableWithGivenFeats(keepSet, '_'.join([x[:3] for x in outcomes]), featNormTable)

    def getCorrelFieldType(self, correlField):
        if correlField == 'state':
            return 'char(2)'
        return None

    def createFeatureTable(self, featureName, featureType = 'VARCHAR(64)', valueType = 'INTEGER', tableName = None, valueFunc = None, correlField=None, extension = None):
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
        sql = """SELECT column_type FROM information_schema.columns WHERE table_schema='%s' AND table_name='%s' AND column_name='%s'""" % (
            self.corpdb, self.corptable, self.correl_field)

        correlField = self.getCorrelFieldType(self.correl_field) if not correlField else correlField
        correl_fieldType = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)[0][0] if not correlField else correlField

        #create sql
        drop = """DROP TABLE IF EXISTS %s""" % tableName
        # featureType = "VARCHAR(30)" # MAARTEN
        #CREATE TABLE feat_3gram_messages_rand1000_user_id (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, user_id ('bigint(20) unsigned',), 3gram VARCHAR(64), VALUE INTEGER
        #sql = """CREATE TABLE %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id %s, feat %s, value %s, group_norm DOUBLE, feat_norm DOUBLE, KEY `correl_field` (`group_id`), KEY `feature` (`feat`)) CHARACTER SET utf8 COLLATE utf8_general_ci""" %(tableName, correl_fieldType, featureType, valueType)
        sql = """CREATE TABLE %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id %s, feat %s, value %s, group_norm DOUBLE, KEY `correl_field` (`group_id`), KEY `feature` (`feat`)) CHARACTER SET %s COLLATE %s""" %(tableName, correl_fieldType, featureType, valueType, self.encoding, fwc.DEF_COLLATIONS[self.encoding.lower()])

        #run sql
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

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
                fwc.warn("getting distinct values for controls")
                controlValuesToAvg = outcomeGetter.getDistinctOutcomeValues(outcome = controlField, includeNull = False, where=outcomeRestriction)

        #create new table name:
        nameParts = featureTable.split('$')
        nameParts = map(lambda part: part.replace('16to', ''), nameParts)
        nameParts = map(lambda part: part.replace('messages', 'msgs'), nameParts)
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
            mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
            sql = 'ALTER TABLE %s ADD COLUMN `N` int(16) not null default -1'%(newTable)
            mm.execute(self.corpdb, self.dbCursor, sql)
            sql = 'ALTER TABLE %s CHANGE feat_norm std_dev FLOAT' % newTable;
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

            
        outres = outcomeRestriction
        outres = outres + ' AND ' if outres else '' #only need and if it exists
        if controlField:
            for outcomeValue, cntrlcounts in \
                    outcomeGetter.getDistinctOutcomeAndControlValueCounts(control = controlField, includeNull = False, where=outcomeRestriction).iteritems():
                for cvalue, count in cntrlcounts.iteritems():
                    if cvalue in controlValuesToAvg:
                        newTable = 'feat_grpd'+ nameSuffix + '$' + '$'.join(nameParts[1:3]) + '$' + outcomeField + '_' + controlField + '_' + str(cvalue) + '$' + '$'.join(nameParts[4:])
                        print "on %s %s and %s %s, count: %d" % (outcomeField, str(outcomeValue), controlField, str(cvalue), count)
                        sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) SELECT age, feat, total_freq, mean_rel_freq, SQRT((N_no_zero*(POW((mean_no_zero - mean_rel_freq), 2) + std_no_zero*std_no_zero) + (N - N_no_zero)*(mean_rel_freq * mean_rel_freq)) / N) as std, N  from (SELECT b.%s, feat, SUM(value) as total_freq, SUM(group_norm)/%d as mean_rel_freq, AVG(group_norm) as mean_no_zero, std(group_norm) as std_no_zero, %d as N, count(*) as N_no_zero FROM %s AS a, %s AS b WHERE %s b.%s = '%s' AND b.%s = '%s' AND b.user_id = a.group_id group by b.%s, a.feat) as stats" % (newTable, outcomeField, count, count, featureTable, outcomeTable, outres, controlField, str(cvalue), outcomeField, str(outcomeValue), outcomeField)
#SELECT age, feat, total_freq, mean_rel_freq, SQRT((N_no_zero*(POW((mean_no_zero - mean_rel_freq), 2) + std_no_zero*std_no_zero) + (N - N_no_zero)*(mean_rel_freq * mean_rel_freq)) / N) as std, N  from (
#SELECT b.age, feat, SUM(value) as total_freq, SUM(group_norm)/390 as mean_rel_freq, AVG(group_norm) as mean_no_zero, std(group_norm) as std_no_zero, 390 as N, count(*) as N_no_zero FROM feat$1gram$messages_en$user_id$16to16$0_01 AS a, masterstats_andy AS b WHERE UWT >= 1000 AND b.age = '45' AND b.user_id = a.group_id group by b.age, a.feat) as a             
                        mm.execute(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)
                    else:
                        print "skipping %s %s and %s %s, count: %d because control value not in list" % (outcomeField, str(outcomeValue), controlField, str(cvalue), count)
        else: #no controls to avg
            # Maarten
            correspondences = outcomeGetter.getGroupAndOutcomeValues()
            correspondences_inv = {}
            for k,v in correspondences:
                correspondences_inv[v] = correspondences_inv.get(v,[])
                correspondences_inv[v].append(k)
            correspondences = correspondences_inv
            total_sum_values = {i[0]: long(i[1]) for i in self.getSumValuesByGroup()}

            i = 0
            j = 0
            for outcomeValue, groups in correspondences.iteritems():
                i += 1
                rows = []
                groups_nonZero = [g for g in groups if g in total_sum_values]
                for feat, values, gns, Nfeats in self.yieldGroupNormsWithZerosByFeat(groups = groups, values = True):
                    if not values: continue

                    sum_value = sum(values.values())
                    total_sum_value = sum(total_sum_values[g] for g in groups_nonZero)
                    group_norm = float(sum_value)/total_sum_value
                    std_dev = std(gns.values())
                    N = len(gns)
                    rows.append([outcomeValue, feat, sum_value, group_norm, std_dev, N])
                    if len(rows) >= 10000:
                        sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) " % newTable
                        sql += "VALUES (%s)" % ', '.join('%s' for r in rows[0]) 
                        mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                        j += len(rows)
                        print "    wrote %d rows [finished %d outcome_values]" % (j, i)
                        rows = []
                    
                if rows:
                    sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) " % newTable
                    sql += "VALUES (%s)" % ', '.join('%s' for r in rows[0])
                    mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                    j += len(rows)  
                    print "    wrote %d rows [finished %d outcome_values]" % (j, i)
                print "Inserted into %s" % newTable

                        
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
            mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

            #create insert fields:
            shortNames = map(lambda i: chr(ord('a')+i), range(len(newTables)))
            tableNames = ', '.join(map(lambda i: "%s as %s" % (newTables[i], shortNames[i]), range(len(newTables))))
            values = "(%s)" % (' + '.join(map(lambda name: "%s.value" % (name), shortNames))) + ' / ' + str(len(shortNames))
            groupNorms = "(%s)" % (' + '.join(map(lambda name: "%s.group_norm" % (name), shortNames))) + ' / ' + str(len(shortNames))
            Ns = ' + '.join(map(lambda name: "%s.N" % (name), shortNames))
            stdDev = "(%s)" % (' + '.join(map(lambda name: "POW(%s.group_norm - (%s), 2) + POW(%s.std_dev, 2)" % (name, groupNorms, name), shortNames))) + ' / ' + str(len(shortNames))
            stdDev = "SQRT(%s)" % stdDev
            #stdDev = "SQRT(%s)" % (' + '.join(map(lambda name: "%s.N*(POW(%s.group_norm - %s, 2) + POW(%s.std_dev,2))" % (name, name, groupNorms, name), shortNames))) + ' / ' + Ns
            #stdDev = "(%s)" % (' + '.join(map(lambda name: "%s.std_dev_no_zero" % (name), shortNames))) + ' / ' + str(len(shortNames))


            #create joins
            groupIds = map(lambda name: "%s.group_id" % (name), shortNames)
            feats = map(lambda name: "%s.feat" % (name), shortNames)
            groupIdJoins = []
            featJoins = []
            for i in xrange(len(groupIds) - 1):
                groupIdJoins.append('%s = %s' % (groupIds[i], groupIds[i+1]))
                featJoins.append('%s = %s' % (feats[i], feats[i+1]))
            groupIdJoins = ' AND '.join(groupIdJoins)
            featJoins = ' AND '.join(featJoins)

            #call SQL
            sql = "INSERT INTO %s (group_id, feat, value, group_norm, std_dev, N) SELECT a.group_id, a.feat, %s, %s, %s, %s FROM %s where %s AND %s" % \
                (avgTable, values, groupNorms, stdDev, Ns, tableNames, groupIdJoins, featJoins)
            print "Populating AVG table with command: %s" % sql
            mm.execute(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)


    def createAggregateFeatTableByGroup(self, valueFunc = lambda d: d):
        """combines feature tables, and groups by the given group field"""
        
        featureTable = self.featureTable

        (_, name, oldCorpTable, oldGroupField) = featureTable.split('$')[:4]
        theRest = featureTable.split('$')[4:]

        

        newTable = 'feat$agg_'+name[:12]+'$'+oldCorpTable+'$'+self.correl_field # +'$'+'$'.join(theRest)
        drop = """DROP TABLE IF EXISTS %s""" % (newTable)
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)

        sql = """CREATE TABLE %s like %s""" % (newTable, featureTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = """ALTER TABLE %s MODIFY group_id VARCHAR(255)""" % (newTable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        

        mm.disableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode)
        
        fwc.warn("Inserting group_id, feat, and values")
        sql = "INSERT INTO %s SELECT m.%s, f.feat, sum(f.value), 0 FROM %s AS f, %s AS m where m.%s = f.group_id GROUP BY m.%s, f.feat" % (newTable,self.correl_field, featureTable, self.corptable, oldGroupField, self.correl_field)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

        fwc.warn("Recalculating group_norms")
        sql = "UPDATE %s a INNER JOIN (SELECT group_id,sum(value) sum FROM %s GROUP BY group_id) b ON a.group_id=b.group_id SET a.group_norm=a.value/b.sum" % (newTable,newTable)
        
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
  
        # patrick changed this to be all SQL 7/21/15. Values and group norms were being calculated wrong before

        fwc.warn("Done inserting.\nEnabling keys.")
        mm.enableTableKeys(self.corpdb, self.dbCursor, newTable, charset=self.encoding, use_unicode=self.use_unicode)
        fwc.warn("done.")

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
        N = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)[0][0]

        feat_counts = self.getFeatureCounts() #tuples of: feat, count (number of groups feature appears with)

        fwc.warn('Inserting idf values into new table')
        counter = 0
        for (feat, dt) in feat_counts:
            idf = log(N/float(dt))

            # get (group_id, group_norm) where feat = feat
            # clean_feat = mm.MySQLdb.escape_string(feat.encode('utf-8')) 

            sql = u"""SELECT group_id, value, group_norm from %s WHERE feat = \'%s\'"""%(ngram_table, mm.MySQLdb.escape_string(feat.encode('utf-8')).decode('utf-8'))

            group_id_freq = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

            for (group_id, value, tf) in group_id_freq:
                tf_idf = tf * idf

                insert_sql = u"INSERT INTO {} (group_id, feat, value, group_norm) VALUES (\'{}\', \'{}\', {}, {});".format(
                                                idf_table, 
                                                group_id, 
                                                mm.MySQLdb.escape_string(feat.encode('utf-8')).decode('utf-8'), 
                                                value, 
                                                tf_idf)
                mm.execute(self.corpdb, self.dbCursor, insert_sql)

                if (counter % 50000 == 0):
                    print '%d tf_idf values inserted!' % (counter)
                counter += 1

        fwc.warn('Finished inserting.')

        return idf_table
