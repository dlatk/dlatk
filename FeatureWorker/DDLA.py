###############################
## correlates results between two csvs of results
## csvs in arguments:
## test-retest.rmatrix.py <rmatrix_csv> <rmatrix_csv>

import csv
import sys
from scipy.stats.stats import pearsonr, spearmanr
from pprint import pprint
from numpy import array, concatenate, log, sqrt, isnan, isinf,  std, clip, nan_to_num

#csv.quotechar = '\x07'
csv.field_size_limit(sys.maxsize)
class DDLA:

    ignoreColumns = set(['p', 'N', 'freq', 'feature'])
    file1 = None
    file2 = None
    data = None
    header = None
    outputData = None
    
    def __init__(self, file1, file2, outputFile = None):
        self.file1 = file1
        self.file2 = file2
        self.outputFile = outputFile if outputFile else '-'.join([file1[:-4], file2[:-4]])+'.csv'
        self.data = list()
        self.header = list()
    
    def signed_r_square(self,r):
        r2 = float(r)**2
        if r < 0: 
            return -1 * r2
        return r2

    def signed_r_log(self,r):
        if r < 0: 
            return -1 * log((-1*float(r))+1)
        return log(r+1)

    def compare_correl(self,ra, na, rb, nb):
        if any(not isnan(i) for i in [ra, na, rb, nb]):
            (ra, rb) = min(1, max(-1, ra)), min(1, max(-1, rb))
            raplus = 1*ra+1
            raminus = 1-ra
            rbplus = 1*rb+1
            rbminus = 1-rb

            za = (log(raplus)-log(raminus))/2
            zb = (log(rbplus)-log(rbminus))/2

            se = sqrt((1/(na-3))+(1/(nb-3)))
            z  = (za-zb)/se

            z2 = abs(z)

            p2 = (((((.000005383*z2+.0000488906)*z2+.0000380036)*z2+.0032776263)*z2+.0211410061)*z2+.049867347)*z2+1

            p2 = pow(p2, -16)

            return p2

        else:
            return float('nan')

    def write2CSV(self, dataDict, features):
        toWrite = dataDict['data'].tolist()
        for i in xrange(len(features)): toWrite[i].insert(0,features[i])
        toWrite.sort(key=lambda x: x[0])
        self.outputData = toWrite
        with open(self.outputFile,'w+') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(dataDict['header'])
            write.writerows(toWrite)

    def add2Output(self,csvOutput, outcome_data, outcome_name, featsInOrder):
        data = self.data
        ordered = ['value', 'p', 'freq', 'N']
        originalData0 = array([[data[0][outcome_name][feat][col] for col in ordered]
                               for feat in featsInOrder])
        originalData1 = array([[data[1][outcome_name][feat][col] for col in ordered]
                               for feat in featsInOrder])
        originalData = concatenate((originalData0,originalData1), axis=1)
        csvOutput['header'].extend([outcome_name, 'p'] + ['r_0', 'p_0', 'freq_0', 'N_0'] + ['r_1', 'p_1', 'freq_1', 'N_1'])
        self.header = csvOutput['header']

        if 'data' in csvOutput.keys():
            csvOutput['data'] = concatenate((csvOutput['data'], nan_to_num(outcome_data)), axis = 1)
        else:
            csvOutput['data'] = nan_to_num(outcome_data)
        csvOutput['data'] = concatenate((csvOutput['data'], originalData),axis=1)

    def get_next(self,some_iterable, window=1):
        from itertools import tee, islice, izip_longest
        items, nexts = tee(some_iterable, 2)
        nexts = islice(nexts, window, None)
        return izip_longest(items, nexts)

    def print_sorted(self, dataDict, features):
        data = dataDict['data'].tolist()
        for i in xrange(len(features)): data[i].insert(0,features[i])
        data.sort(key=lambda x: -x[1])

    def load_data(self):
        data = self.data
        fins = [self.file1, self.file2]
        for split in xrange(len(fins)):
            fin = open(fins[split], 'r')
            reader = csv.reader(fin)
            data.append(dict())

            #get headers:
            headers = None
            while True:
                headers = reader.next()
                if len(headers) > 1 and headers[0][:6] != 'Namesp':
                    #remove blanks and namespace lines
                    break

            #setup data dict, using column names
            for h in headers:
                h = h.strip()
                if h not in (self.ignoreColumns):
                    data[split][h] = dict()

            #read all rows:
            for row in reader:
                if row and row[0] == 'SORTED:':
                    print "found SORTED:", row
                    break
                if len(row) > 1:
                    feat = row[0].strip()
                    column_used = None
                    for i in xrange(1, len(row)):
                        # Going through the entire row, entry by entry
                        if i < len(headers):
                            column_name = None
                            if headers[i] in data[split]:
                                # New Outcome
                                column_used = headers[i]
                                data[split][column_used][feat] = dict()
                                column_name = 'value'
                            else:
                                column_name = headers[i]
                            cell_value = row[i].strip()
                            if cell_value :#and not isnan(float(cell_value)):
                                data[split][column_used][feat][column_name] = float(cell_value)

    def outputForTagclouds(self, sizeField = 1, colorField = "dr"):
        correls = dict()
        outcomes = self.header[1::10]
        correls = {outcome: dict() for outcome in outcomes}
        # print self.header[1::10] # color
        # print self.header[sizeField+2::10] # first country correlations = size
        # print self.header[sizeField+6::10] # secnd country correlations = size
        for row in self.outputData:
            drs = row[1::10]
            rs = row[sizeField+2::10]
            ps = row[sizeField+3::10]
            Ns = row[sizeField+5::10]
            for i,o in enumerate(outcomes):
                correls[o][row[0]] = (rs[i], ps[i], Ns[i], drs[i])
        return correls

    def differential(self):
        data = self.data
        self.load_data()
        #from pprint import pprint
        #pprint([(k, v) for k,v in data[1]['is_student'].iteritems() if k == 'den'][:10])
        commonOutcomes = set(data[0].keys()) & set(data[1].keys()) 

        sumRs = float(0)
        sumR2s = float(0)
        sumRhos = float(0)
        sumRho2s = float(0)
        csvOutput = {'header': ['feature',]}
        commonFeats = None

        for outcome in sorted(commonOutcomes):
            print "\n%s\n%s" % (outcome, '='*len(outcome))
            feats0 = set(data[0][outcome].keys())
            feats1 = set(data[1][outcome].keys())
            commonFeats = feats0 & feats1
            print "Number of feats in first results:  %d" % len(feats0)
            print "Number of feats in second results: %d" % len(feats1)
            print "Number of feats in common:         %d" % len(commonFeats)

            # Getting data in the right format
            commonFeats = list(commonFeats)
            list0 = nan_to_num(array([data[0][outcome][feat]['value'] for feat in commonFeats]))
            list1 = nan_to_num(array([data[1][outcome][feat]['value'] for feat in commonFeats]))
            list0_n = array([data[0][outcome][feat]['N'] for feat in commonFeats])
            list1_n = array([data[1][outcome][feat]['N'] for feat in commonFeats])

            # Comparing individual correlations
            diffs = list0 - list1
            output = array([[diffs[i], self.compare_correl(list0[i], list0_n[i], list1[i], list1_n[i])] for i in xrange(len(list0))])
            self.add2Output(csvOutput, output, outcome, commonFeats)

            sorted_pairs = sorted(zip(diffs, commonFeats), key= lambda (x,y): -x)
            sorted_words = map(lambda (x,y): y, sorted_pairs)
            print 'Decreasing differences: <top 10 most correlated with %s> ... <top 10 most correlated with %s>' % (self.file1[:-4], self.file2[:-4] )
            print ', '.join(sorted_words[:10])+' ... '+', '.join(sorted_words[-10:])


            # From previous script, comparing all the r's
            list0r2 = array([self.signed_r_log(r) for r in list0])
            list1r2 = array([self.signed_r_log(r) for r in list1])
            (r, p) = pearsonr(list0, list1)
            (r2, p2) = pearsonr(list0r2, list1r2)
            print "pearson r of rs:               %10.4f (%.6f)" % (r, p)
            print "pearson r of signed log(r)s:   %10.4f (%.6f)" % (r2, p2)
            sumRs += r
            sumR2s += r2
            (rho, p) = spearmanr(list0, list1)
            (rho2, p2) = spearmanr(list0r2, list1r2)
            print "spearman rho of rs:            %10.4f (%.6f)" % (rho, p)
            print "spearman rho of signed log(r)s:%10.4f (%.6f)" % (rho2, p2)
            sumRhos += rho
            sumRho2s += rho2

        self.outputData = csvOutput
        self.write2CSV(csvOutput, commonFeats)
        # print_sorted(csvOutput, commonFeats)


        print "\nAVERAGE RESULTS\n==============="
        print "pearson r of rs:               %10.4f " % (sumRs / float(len(commonOutcomes)))
        print "pearson r of log(r)s:          %10.4f " % (sumR2s / float(len(commonOutcomes)))
        print "spearman rho of rs:            %10.4f " % (sumRhos / float(len(commonOutcomes)))
        print "spearman rho of log(r)s:       %10.4f " % (sumRho2s / float(len(commonOutcomes)))

#### TODO: calculate number in common in top 100 (i.e. as if comparing word clouds)

if __name__ == '__main__':
    fins = sys.argv[1:3]
    ddla = DDLA(fins[0], fins[1])
    print dir(ddla)
    ddla.differential()
