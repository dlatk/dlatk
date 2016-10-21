#!/usr/bin/python

__author__ = "Lukasz Dziurzynski"
__copyright__ = "Copyright 2012"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "0.1"
__maintainer__ = "Lukasz Dziurzynski"
__email__ = "lukaszdz@sas.upenn.edu"


import sys
import time
import argparse
import MySQLdb
import re
import pickle
from pprint import pprint
from math import sqrt, log, isnan
from numpy import array, mean, std, var, isnan, zeros
import numpy.random as rand
from scipy.stats.stats import pearsonr
from scipy.stats import scoreatpercentile
from math import floor, ceil, log
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.rlike.container as rlc
from wwbp import notify
import wwbp.mysqlMethods as mm
import array
from numpy import linspace
try:
    import rpy2.interactive 
    ro.r.library('gregmisc')
except ImportError:
    fwc.warn("rpy2 cannot be imported")
    pass

#DB INFO:
HOST = '127.0.0.1'
USER = os.getlogin()
PASSWD = ''

#plotting:
N_HIST_BREAKS = 16
#http://research.stowers-institute.org/efg/R/Graphics/Basics/mar-oma/index.htm
#http://www.r-bloggers.com/setting-graph-margins-in-r-using-the-par-function-and-lots-of-cow-milk/
#rpy2.interactive.packages.utils.install_packages('gregmisc')

class StatsPlotter(object):
    def __init__(self, heights=700, widths=700):
        self.grdevices = importr('grDevices')
        self.graphics = importr('graphics')
        self.stats = importr('stats')
        self.heights = heights
        self.widths = widths

    def plotDescStats(self, dataDict, filename=None, n_breaks=None, invValueFunc=None):
        """Plot boxplots and histograms for the specified dataDict.
           dataDict maps a string (column name) to a list of its values.
           Returns a dictionary containing descriptive statistics."""
        descStatdict = dict()
        for col in dataDict:
            colDescStats = dict()
            data = [x for x in dataDict[col] if x != None]
            data = list(map(float, data))
            total_count = len(dataDict[col])
            if data:
                colDescStats['avg'] = mean(data)
                colDescStats['count'] = len(data)
                colDescStats['std'] = std(data)
                colDescStats['min'] = max(data)
                colDescStats['max'] = min(data)
                ro_data = ro.FloatVector(data)
                ro_breaks = 'FD'
                if n_breaks:
                    ro_breaks = ro.FloatVector(list(linspace(min(data), max(data), n_breaks, True)))
                if filename:
                    self.grdevices.png(file="%s_%s_box.png"%(filename, col), width=self.widths, height=self.heights)
                    #ro.r.boxplot(ro_data, main="%s n=%d, N=%d"%(col, len(data), total_count), col='royalblue4', xlab="value range (in [%2.2f, %2.2f])"%(min(data), max(data)))
                    ro.r.boxplot(ro_data, main="%s N=%d/%d"%(col, len(data), total_count), col='royalblue4', xlab="value range (in [%2.2f, %2.2f])"%(min(data), max(data)))
                    self.grdevices.dev_off()
                    self.grdevices.png(file="%s_%s_hist.png"%(filename, col), width=self.widths, height=self.heights)
                    xaxt = 's'
                    if invValueFunc:
                        xaxt='n'
                    #ro.r.hist(ro_data, breaks=ro_breaks, col='royalblue4', main="%s n=%d, N=%d"%(col, len(data), total_count), xlab="value range (in [%2.2f, %2.2f])"%(min(data), max(data)))
                    ro.r.hist(ro_data, breaks=ro_breaks, col='royalblue4', main="%s N=%d/%d"%(col, len(data), total_count), xlab="value range (in [%2.2f, %2.2f])"%(min(data), max(data)), xaxt=xaxt)
                    if invValueFunc:
                        ro.r.axis(1, at=ro.r.axTicks(1), labels=ro.FloatVector(list(map(invValueFunc, list(ro.r.axTicks(1))))))
                    self.grdevices.dev_off()
            descStatdict[col] = colDescStats
        return descStatdict

    def plotPie(self, dataDict, filename=None, colors=None, labels=None, startAngle=0):
        """Plot a pie chart for the specified dataDict.
           dataDict maps a string (column name) to a list of its values."""
        for col in dataDict:
            data = [x for x in dataDict[col] if x != None]
            data = list(map(float, data))
            #total_count = len(dataDict[col])
            if data:
                ro_data = ro.StrVector(data)
                ro_frequencies = ro.r.table(ro_data)
                ro_percentages = ro.FloatVector( ro.r.round( (ro_frequencies.ro / ro.r.sum(ro_frequencies)).ro * 100.0, 2) )
                ro_labels = ro.r.names(ro_frequencies)
                ro_colors = self.grdevices.rainbow(len(ro_frequencies))
                if labels:
                    ro_labels = ro.StrVector(labels)
                ro_labels = ro.r.paste(ro_labels, ' ', ro_percentages, '%', sep="")
                if colors:
                    ro_colors = ro.StrVector(colors)
                if filename:
                    self.grdevices.png(file="%s_%s_pie.png"%(filename, col), width=self.widths, height=self.heights)
                    ro.r.pie(ro_frequencies, main="%s N=%d"%(col, len(data)), col=ro_colors, labels=ro_labels, **{'init.angle':startAngle } )
                    self.grdevices.dev_off()        

    def plotScatter(self, x_name, x_values, y_name, y_values, filename=None):
        """Make a scatterplot"""
        ro_x = ro.FloatVector(x_values)
        ro_y = ro.FloatVector(y_values)
        if filename:
            self.grdevices.png(file="%s.png"%(filename), width=self.widths, height=self.heights)
        else:
            self.grdevices.png(file="scat_%s-%s.png"%(x_name, y_name), width=self.widths, height=self.heights)
        ro.globalenv['x'] = ro_x
        ro.globalenv['y'] = ro_y
        ro.r('regr <- lm(y~x)')
        intercept = ro.r('int <- regr$coefficients[1]')[0]
        beta = ro.r('beta <- regr$coefficients[2]')[0]
        r_squared = ro.r('r_squared <- summary(regr)$r.squared')[0]
        ro.r.plot(ro_x, ro_y, xlab=x_name, ylab=y_name, main="Y ~ %.4g + %.4g x   r2: %2.4f"%(intercept, beta, r_squared))
        ro.r('abline(regr, col="red")')
        if filename:
            self.grdevices.dev_off()

    def _getBins(self, value_list):
        n = len(value_list)
        if n < 200:
            return ceil( log(n,2) + 1 )
        else:
            p75 = scoreatpercentile(value_list, 75)
            p25 = scoreatpercentile(value_list, 25)
            return ceil( (max(value_list)-min(value_list)) / (2.0*(p75-p25)*(n**(-1.0/3.0))) )
        

    def plot2dHist(self, x_name, x_values, y_name, y_values, filename=None):
        """Plot the 2d histogram from gregmisc R package"""
        ro_x = ro.FloatVector(x_values)
        ro_y = ro.FloatVector(y_values)
        if filename:
            self.grdevices.png(file="%s.png"%(filename), width=self.widths, height=self.heights)
        else:
            self.grdevices.png(file="2dHist_%s-%s.png"%(x_name, y_name), width=self.widths, height=self.heights)
        ro.globalenv['x'] = ro_x
        ro.globalenv['y'] = ro_y
        ro.r('regr <- lm(y~x)')
        intercept = ro.r('int <- regr$coefficients[1]')[0]
        beta = ro.r('beta <- regr$coefficients[2]')[0]
        r_squared = ro.r('r_squared <- summary(regr)$r.squared')[0]
        n_bins = self._getBins(x_values) * self._getBins(y_values)
        rgb_palette = self.grdevices.colorRampPalette(ro.StrVector(['lightyellow', 'orange']), space='rgb')
        ro_colors = ro.StrVector(['white'] + list(rgb_palette(20)))
        ro.r.hist2d(ro_x, ro_y, xlab=x_name, ylab=y_name, main="Y ~ %.4g + %.4g x   r2: %2.4f"%(intercept, beta, r_squared), col=ro_colors, nbins=n_bins)
        ro.r('abline(regr, col="black")')
        if filename:
            self.grdevices.dev_off()

    def plot2dHistGeneralized(self, names_to_x, names_to_y, filename=None):
        """Plot the 2d histogram for all combinations of x and y"""
        n_rows = len(names_to_x)
        n_cols = len(names_to_y)

        if filename:
            self.grdevices.png(file="%s_%s_%s_2dhist.png"%(filename, '-'.join(list(names_to_x.keys())), '-'.join(list(names_to_y.keys())) ), width=200*n_rows, height=400*n_cols)
            ro.r.par(mfrow=ro.IntVector([n_rows, n_cols]))

        for x_name in names_to_x:
            for y_name in names_to_y:
                self.plot2dHist(x_name, names_to_x[x_name], y_name, names_to_y[y_name])
                
        if filename:
            self.grdevices.dev_off()

    def getFloatColStats(self, db, dbCursor, tableName, colsOfNote, filename=None):
        #1. Get column names
        sql = "SELECT column_name from information_schema.columns where table_name='%s'"%tableName
        row = mm.executeGetList(db, dbCursor, sql)
        colNames = []
        for col in colsOfNote:
            colNames.append(row[col][0])

        ncols = len(colsOfNote)

        #. Get users who have at least one message
        sql = "SELECT DISTINCT group_id FROM feat$1gram$messages$user_id$16to16"
        rows = mm.executeGetList(db, dbCursor, sql)
        user_ids = []
        for row in rows:
            user_ids.append(row[0])
        sql_user_ids = ",".join(map(str,user_ids))

        #. Get total row count
        total_count = len(user_ids)

        #. Assemble storage data structure
        dataHolder = []
        for col in colsOfNote:
            dataHolder.append([None]*total_count)

        #. Pull data of interest; use offset if needed
        sql = "SELECT * FROM %s WHERE user_id IN (%s)"%(tableName, sql_user_ids)
        rows = mm.executeGetList(db, dbCursor, sql)
        ii = 0
        for row in rows:
            jj = 0
            for col in colsOfNote:
                dataHolder[jj][ii] = row[col]
                jj += 1
            ii += 1

        #. Link the data to their names
        dataDict = dict()
        for cc in range(ncols):
            dataDict[colNames[cc]] = dataHolder[cc]

        return self.plotDescStats(dataDict, total_count, filename)
        
    def getCategoricalColStats(self, db, dbCursor, tableName, colsOfNote, filename=None):
        #1. Get column names
        sql = "SELECT column_name from information_schema.columns where table_name='%s'"%tableName
        row = mm.executeGetList(db, dbCursor, sql)
        colNames = []
        for col in colsOfNote:
            colNames.append(row[col][0])

        ncols = len(colsOfNote)

        #. Get users who have at least one message
        sql = "SELECT DISTINCT group_id FROM feat$1gram$messages$user_id$16to16"
        rows = mm.executeGetList(db, dbCursor, sql)
        user_ids = []
        for row in rows:
            user_ids.append(row[0])
        sql_user_ids = ",".join(map(str,user_ids))

        #. Get total row count
        total_count = len(user_ids)

        #. Assemble storage data structure
        dataHolder = dict()
        for col in colNames:
            dataHolder[col] = 0
        dataHolder["none specified"] = 0

        #. Pull data of interest
        sql = "SELECT * FROM %s WHERE user_id IN (%s)"%(tableName, sql_user_ids)
        rows = mm.executeGetList(db, dbCursor, sql)
        ii = 0
        for row in rows:
            jj = 0
            has_specified_value = False
            for col in colsOfNote:
                if row[col]:
                    dataHolder[colNames[jj]] += 1
                    if has_specified_value:
                        raise Exception("incorrect assumption; more than one category allowed")
                    has_specified_value = True
                jj += 1
            if not has_specified_value:
                dataHolder["none specified"] += 1
            ii += 1

        #. Calculate descriptive statistics and create plots
        labels = list(dataHolder.keys())
        counts = list(dataHolder.values())
        ro_labels = ro.StrVector(labels)
        ro_counts = ro.IntVector(counts)
        if filename:
            self.grdevices.png(file="%s_hist_cats.png"%(filename), width=self.widths, height=self.heights)
            self.graphics.par(las=2, mar=[5.1, 7.1, 4.1, 2.1])
            ro.r.barplot(ro_counts, main = "Category Histogram, N=%d"%(total_count), beside=True, horiz=True, col='royalblue4', **{"names.arg":ro_labels})
            self.grdevices.dev_off()

if __name__=="__main__":
    sp = StatsPlotter()
    #floatCols = [2, 3, 4] + range(6,14) + [23] + [28]
    #prefix = '600_'
    #sp.getFloatColStats("userstats_en", floatCols, "plots/%sdesc"%prefix)
    #sp.getCategoricalColStats("userstats_en", range(14, 23), "plots/%srelnbins"%prefix)
    (conn, cur, dcur) = mm.dbConnect('fb20')
    #sp.getCategoricalColStats('fb20', cur, "userstats_en",  range(24, 28), '/data/ml/plots/fb20/age_category')
    #sp.getCategoricalColStats('fb20', cur, "userstats_en",  range(14, 23), '/data/ml/plots/fb20/reln_category')
    # N=100000
    # d1 = list(rand.normal(0,2,N))
    # d2 = list(rand.normal(0,1,N))
    # d3 = list(rand.normal(0,17,N))
    # d_all = {"d1":d1, "d2":d2, "d3":d3}
    # e1 = list(rand.exponential(2,N))
    # e2 = list(rand.exponential(14,N))
    # e_all = {'e1':e1, 'e2':e2}
    # sp.plot2dHist('d1', d1, 'd2', d2)
    # sp.plot2dHistGeneralized(d_all, e_all, 'plots/samba')
    mm.warn("descStats.py exits with success :)")
