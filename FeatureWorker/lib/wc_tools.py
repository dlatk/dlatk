#wc_tools.py
#for sizing word lists and getting color_lists

from numpy import array, round
from math import log, sqrt
from random import random
from scipy.stats import rankdata
import sys
#infrastructure

sys.path.insert(0,'../') #assume that tools and CoreInfrastructure are in the same directory
sys.path.insert(0,'../../')
sys.path.insert(0,'../mysqlMethods') 

import fwConstants
import argparse
import mysqlMethods as mm

#testing gitolite

def rgbColorMix(fromColor, toColor, resolution, randomness = False):
    #fromColor, toColor rgb (255 max) tuple
    #resolution, how many truple to return inbetween
    #(starts at from, but comes up one short of ToColor)
    (fromColor, toColor) = (array(fromColor), array(toColor))
    fromTo = toColor - fromColor #how to get from fromColor to toColor
    fromToInc = fromTo / float(resolution)
    gradientColors = []
    for i in range(resolution):
        gradientColors.append(tuple([int(x) for x in round(fromColor + (i * fromToInc))]))
    if randomness: 
        for i in range(len(gradientColors)): 
            color = gradientColors[i]
            newcolor = []
            for value in color:
                value += 20 - randint(0, 40)
                value = max(0, min(255, value))
                newcolor.append(value)
            gradientColors[i] = tuple(newcolor)

    #print gradientColors[0:4], gradientColors[-4:] #debug
    return gradientColors

def freqToColor(freq, maxFreq = 1000, resolution=64, colorScheme='multi'):

	#print freq, maxFreq

	perc = freq / float(maxFreq)
	(red, green, blue) = (0, 0, 0)
	if colorScheme=='multi':
	#print "%d %d %.4f" %(freq, maxFreq, perc)#debug
		if perc < 0.17: #grey to darker grey
			(red, green, blue) = rgbColorMix((168, 168, 168),(124, 124, 148), resolution)[int(((1.00-(1-perc))/0.17)*resolution) - 1]
		elif perc >= 0.17 and perc < 0.52: #grey to blue
			(red, green, blue) = rgbColorMix((124, 124, 148), (32, 32, 210), resolution)[int(((0.830-(1-perc))/0.35)*resolution) - 1]
		elif perc >= 0.52 and perc < 0.90: #blue to red
			(red, green, blue) = rgbColorMix((32, 32, 210), (200, 16, 32), resolution)[int(((0.48-(1-perc))/0.38)*resolution) - 1]
		else: #red to dark red
			(red, green, blue) = rgbColorMix((200, 16, 32), (128, 0, 0), resolution)[int(((0.10-(1-perc))/0.10)*resolution) - 1]
	# blue:
	elif colorScheme=='blue':
		if perc <= 0.50: #light blue to med. blue
			(red, green, blue) = rgbColorMix((170, 170, 210), (90, 90, 240), resolution)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
		else: #med. blue to strong blue
			(red, green, blue) = rgbColorMix((90, 90, 240), (30, 30, 140), resolution)[int(((0.5-(1-perc))/0.5)*resolution) - 1]
				# blue:
	elif colorScheme=='old_blue':
		if perc < 0.50: #light blue to med. blue
			(red, green, blue) = rgbColorMix((76, 76, 236), (48, 48, 156), resolution)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
		else: #med. blue to strong blue
			(red, green, blue) = rgbColorMix((48, 48, 156), (0, 0, 110), resolution)[int(((0.5-(1-perc))/0.5)*resolution) - 1]
	#red:
	elif colorScheme=='red': 
		if perc < 0.50: #light red to med. red
			(red, green, blue) = rgbColorMix((236, 76, 76), (156, 48, 48), resolution)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
		else: #med. red to strong red
			(red, green, blue) = rgbColorMix((156, 48, 48), (110, 0, 0), resolution)[int(((0.5-(1-perc))/0.5)*resolution) - 1]
	elif colorScheme=='green': 
		(red, green, blue) = rgbColorMix((166, 247, 178), (27, 122, 26), resolution)[int((1.00-(1-perc))*resolution) - 1]

	elif colorScheme == 'test':
		(red, green, blue) = (255, 255, 255)
	#red+randomness:
	elif colorScheme=='red-random':
		if perc < 0.50: #light blue to med. blue
			(red, green, blue) = rgbColorMix((236, 76, 76), (156, 48, 48), resolution, True)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
		else: #med. blue to strong blue
			(red, green, blue) = rgbColorMix((156, 48, 48), (110, 0, 0), resolution, True)[int(((0.5-(1-perc))/0.5)*resolution) - 1]


	#print "(%d %d %d)" %(red, green, blue)#debug

	htmlcode = "%02s%02s%02s" % (hex(red)[2:], hex(green)[2:], hex(blue)[2:])
	return htmlcode.replace(' ', '0')

def getRankedFreqList(word_list, max_size = 75, min_size = 30, scale = 'linear'):
	# returns freq_list i.e. list of sizes from word_list
	# freq_list goes from biggest to smallest
	# make sure the word_list is sorted accordingly
	
	if len(word_list) == 1:
		return [max_size]

	freq_list = []

	num_blocks = int(log(len(word_list), 2) + 1)
	range = max_size - min_size
	block_size = range/(num_blocks - 1)
	

	i = 1;
	while i <= len(word_list):
		rank = int(log(i, 2))
		#print "{} {}".format(i, rank)

		if scale == 'linear':
			value = max_size - (block_size * rank)
			freq_list.append(value)


		i += 1

	return freq_list

def normalizeFreqList(old_freq_list, word_count = 15):
	'''
	Given a sorted freq_list and a word count, return a normalized freq_list, based on the old sizing algorithm from oa.printTagCloudFromTuples
	:param old_freq_list: list of sorted, descending integers
	:param word_count: an integer that shows how big the new_freq_list should be
	'''

	minR = old_freq_list[-1]
	maxR = old_freq_list[0]
	diff = float(maxR - minR)
	if diff == 0: diff = 0.000001
	smallDataBump = max((word_count - len(old_freq_list)), 10)

	new_freq_list = [int(((freq-minR)/diff)*word_count) + smallDataBump for freq in old_freq_list]

	return new_freq_list



# def getRankList(list):
# 	# i.e. for input [3, 8, 5, 1] returns [1, 3, 2, 0]
# 	rank_list = range(len(list))
# 	list, rank_list = zip(*sorted(zip(list, rank_list)))
# 	return rank_list

def getColorList(word_list, freq_list = [], randomize = False, colorScheme = 'multi', scale = 'linear'):
	color_list = []

	max_freq = 1000
	for i in range(len(word_list)):
		if randomize:
			#print 'Randomizing colors'
			freq = (random() * max_freq) + 1 #a number from 1 to max_freq
			colorHex = freqToColor(freq, maxFreq = max_freq, colorScheme = colorScheme)
			color_list.append(colorHex)
		
		elif freq_list:
			assert (len(word_list) == len(freq_list))
			rank_list = rankdata(freq_list, method = 'ordinal') #, method = 'ordinal'
			#print rank_list
			if scale == 'sqrt':
				max_size = max([sqrt(x) for x in freq_list]) #scaled via sqrt
				freq = sqrt(freq_list[i])

			elif scale == 'linear':
				max_size = len(freq_list)
				freq = rank_list[i]

			#max_size = min([max_size, 100])
			#freq = min(freq_list[i], 100)
			colorHex = freqToColor(freq, maxFreq = max_size, colorScheme = colorScheme)
			color_list.append(colorHex)
			#print '{} {}'.format(word_list[i], max_size)
		else:
			print('Randomize is False, but freq_list is not provided.')



	return color_list

def getFeatValueAndZ(user, schema, ngramTable, min_value = 5, ordered = True, z_threshold = 0):
	#returns list of (feat, value, z) for a given user
	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)

	if ordered:
		order_by = " ORDER BY z DESC"
	else:
		order_by = ""

	pos_z = " AND z > {}".format(z_threshold)


	query = 'SELECT feat, value, z FROM {}.{} WHERE group_id = \'{}\' and value >= {}{}{};'.format(schema, ngramTable, user, min_value, pos_z, order_by)
	print(query)
	list = mm.executeGetList(schema, dbCursor, query)
	#return map(lambda x: x[0], list)
	return list


def getMeanAndStd(word, ngramTable, schema, num_groups = -1, distTable = '', distTableSource = None):
	# get mean and std for a word using the ngramTable

	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)

	if num_groups == -1:
		query = 'SELECT count(distinct(group_id)) FROM {}.{}'.format(schema, ngramTable)
		result = mm.executeGetList(schema, dbCursor, query)
		num_groups = int(result[0][0])
		#print int(num_groups[0][0])

	elif distTableSource is not None:
		#TODO: let user specify distTableSource		
		query = 'SELECT count(distinct(group_id)) FROM {}.{}'.format(schema, distTableSource)
		result = mm.executeGetList(schema, dbCursor, query)
		num_groups = int(result[0][0])
	else:
		pass

	if distTable == '':

		########### two pass algorithm
		n = 0 #count
		sum = 0.0
		diff_squared_sum = 0.0
		#for group_norm in session.query(Feature.group_norm).filter(Feature.feat == word):
		query = 'SELECT group_norm FROM {}.{} WHERE feat = \'{}\''.format(schema, ngramTable, word)
		group_norms = mm.executeGetList(schema, dbCursor, query)
		#print 'SELECT group_norm FROM {}.{} WHERE feat = \'{}\''.format(schema, ngramTable, word)
		num_groups = len(group_norms)

		if len(group_norms) == 1:
			return (0, 0)

		for group_norm in group_norms:
			n += 1
			sum += group_norm[0]


		mean = float(sum)/num_groups
		#print "Mean: %.12f" % mean

		for group_norm in group_norms:
			diff_squared_sum += (group_norm[0] - mean) ** 2

		if (num_groups == 1):
			variance = 1
		else:
			variance = diff_squared_sum / (num_groups - 1) #sample variance
		std = sqrt(variance)
		#print "Standard Deviation: %.12f" % std

		########### algorithm end

	else:
		query = "SELECT mean, std FROM {}.{} where feat = \'{}\'".format(schema, distTable, word)
		result = mm.executeGetList(schema, dbCursor, query)
		if not result:
			mean = 0
			std = 0
		else:
			mean = result[0][0]
			std = result[0][1]

	#print (mean, std)
	return (mean, std)

def getNgrams(ngramTable, schema):
	#returns list of ngrams

	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)
	query = "SELECT feat FROM {}.{} GROUP BY feat".format(schema, ngramTable)
	return mm.executeGetList(schema, dbCursor, query)

def getUsers(schema, ngramTable):
	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)
	query = "SELECT distinct(group_id) FROM {}.{};".format(schema, ngramTable)
	return [user[0] for user in mm.executeGetList(schema, dbCursor, query)]

def updateZscore(schema, ngramTable, user = '', use_feat_table = False, 
					distTable = ''):
	# update ngramTable with z-values

	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)

	counter = 0
	if user != '':
		users = [user]

	else:
		users = getUsers(schema, ngramTable)

	for user in users:
		for ngram in [x[0] for x in getNgrams(ngramTable, schema)]:
			if use_feat_table:
				z = getZscore(ngram, user, ngramTable, schema)
			else:
				z = getZscore(ngram, user, ngramTable, schema, distTable = distTable)

			ngram = ngram.replace('\'', '\'\'')

			try :
				query = "UPDATE {}.{} SET z = {} where group_id = \'{}\' and feat=\'{}\'".format(schema, ngramTable, z, user, ngram)

			except UnicodeEncodeError:
				query = "UPDATE {}.{} SET z = 0 where group_id = \'{}\' and feat=\'{}\'".format(schema, ngramTable, user, ngram.encode('utf-8'))		
				

			if counter % 1000 == 0: print(query)
			mm.executeGetList(schema, dbCursor, query)
			counter += 1

def getZscore(word, user, ngramTable, schema, distTable = ''):
	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)
	word = word.replace('\'', '\'\'')

	try:
		query = 'SELECT group_norm FROM {}.{} where group_id = \'{}\' and feat = \'{}\''.format(schema, ngramTable, user, word)
		#print query
	except UnicodeEncodeError:
		return 0


	group_norm = mm.executeGetList(schema, dbCursor, query)

	if not group_norm:
		#print group_norm
		#print 'group_norm is None'
		return 0 
	if isinstance(group_norm, tuple):
		#print group_norm
		group_norm = group_norm[0]

	if isinstance(group_norm, tuple):
		group_norm = group_norm[0]

	(mean, std) = getMeanAndStd(word, ngramTable = ngramTable, schema = schema, distTable = distTable)
	#print type(group_norm)
	if (std == 0):
		return 0
	else:
		return (group_norm - mean)/(std + 0.0)

def createZColumn(schema, ngramTable):
	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)
	query = "ALTER TABLE {}.{} ADD COLUMN z DOUBLE;".format(schema, ngramTable)
	mm.executeGetList(schema, dbCursor, query)

def getOneGram(schema, ngramTable):
	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)
	query = "SELECT feat, sum(value) as count FROM {}.{} group by feat".format(schema, ngramTable)
	print(query)
	return mm.executeGetList(schema, dbCursor, query)

def getUniqueNgrams(schema, ngramTable, user = '', max = -1):
	# get n ngrams from ngramTable where z-score = 0, sorted by group_norm
	# if user is specified, only grab unique ngrams from that user

	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)

	if user != '':
		select_user = ' AND group_id = \'{}\''.format(user)
	else:
		select_user = ''

	if max != -1:
		limit = ' LIMIT {}'.format(max)
	else:
		limit = ''

	query = 'SELECT feat, group_norm FROM {}.{} WHERE z = 0{} ORDER BY group_norm DESC{}'.format(schema, ngramTable, select_user, limit)
	return mm.executeGetList(schema, dbCursor, query)

def getFeatWithLimit(schema, table, group = '', amount = 50, orderBy = 'group_norm', desc = True):
	#get the first n amount of words, using the orderBy (asc or desc) column to sort. 
	#if group is specified, get from that specific group
	#returns list of (feat, group_norm)

	(dbConn, dbCursor, dictCursor) = mm.dbConnect(schema)

	if group != '':
		select_group = 'where group_id = \'{}\''.format(group)
	else:
		select_group = ''

	if amount <= 0:
		limit = ''
	else:
		limit = ' LIMIT {}'.format(int(amount))

	query = 'SELECT feat, group_norm FROM {}.{} {} ORDER BY {} DESC{}'.format(schema, table, select_group, orderBy, limit)
	return mm.executeGetList(schema, dbCursor, query)
