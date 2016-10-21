#!/usr/bin/python
import re
import os
import pdb
import struct
import string
from glob import glob
from random import uniform as runif
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import imp, sys

import subprocess
from subprocess import check_call, CalledProcessError

from PIL import Image
from numpy import array

#try:
#    ro.r.library('Cairo')       #Optional, only used for old wordcloud module
#    ro.r.library('wordcloud')   #Optional, only used for old wordcloud module
#except:
#    print 'R wordcloud library not imported'


#ro.r.library('extrafont')
#ro.r.loadfonts()
#ro.r.font_import()
font_ii = 1
## comment

# requires rpy2, Cairo(rpackage), Wordcloud(rpackage), extrafont(Rpackage), ImageMagick (installed on the system)
# also: sudo apt-get install libcairo2-dev libxt-dev

#GRDEVICES = importr('grDevices')
#brewer = importr('RColorBrewer')

def rgb(hex_str):
    """converts a hex string to an rgb tuple"""
    return struct.unpack('BBB', hex_str.decode('hex'))

"""def rgbToHex(rgb_tup):
    #converts an (R, G, B) tuple to Hex code
    for value in rgb_tup:
        if value > 255:
            _warn("An RGB value above 255 found. Capped.")
            value = 255 #cap value of any RGB value to 255

    r = "%02X" % rgb_tup[0] #convert decimal to hex
    g = "%02X" % rgb_tup[1]
    b = "%02X" % rgb_tup[2]

    print r
    print g
    print b
    return r + g + b"""

def findall(string, sep=':'):
    return [m.start() for m in re.finditer(sep, string)]

def explode(string):
    colons = findall(string)
    first = colons[-2]
    second = colons[-1]
    return (string[:first], string[first+1:second], string[second+1:])

def extract(text, sub1, sub2):
    return text.split(sub1)[-1].split(sub2)[0]


# handle <:) : 10 in this.
def wordcloudByFile(input_filename, output_filename='wordle_test'):
    """Reads a file in format (per line) word:number \n and builds a wordle"""
    word_freq_pairs = []
    f = open(input_filename, 'rb')
    for line in f.readlines():
        word_freq_pairs.append(line.rstrip().split(':'))
    word_list = map(lambda x: x[0], word_freq_pairs)
    freq_list = map(lambda x: x[1], word_freq_pairs)
    wordcloud(word_list, freq_list)

def _makeRandomColorTuples(num_rgb_tuples, rgb_bounds=[0, 0.39]):
    #Create a list of length num_rgb_tuples, where every element is a random hex color
    assert(num_rgb_tuples > 0)
    r = map(runif, [rgb_bounds[0]]*num_rgb_tuples, [rgb_bounds[1]]*num_rgb_tuples)
    g = map(runif, [rgb_bounds[0]]*num_rgb_tuples, [rgb_bounds[1]]*num_rgb_tuples)
    b = map(runif, [rgb_bounds[0]]*num_rgb_tuples, [rgb_bounds[1]]*num_rgb_tuples)
    ro_colors = map(ro.r.rgb, r, g, b)
    ro_color_list = ro.StrVector(map(lambda x: x[0], ro_colors))
    return ro_color_list

def _makeColorTuplesFromRgbTuples(rgbTuples):
    r, g, b = zip(*rgbTuples)
    N = len(r)
    blank_lists = [ [] for l in range(N) ]
    ro_colors = map(ro.r.rgb, r, g, b, [255]*N, blank_lists, [255]*N)
    ro_color_list = ro.StrVector(map(lambda x: x[0], ro_colors))
    return ro_color_list
    

def wordcloud(word_list, freq_list, output_prefix='test', 
    color_list=None, random_colors=True, random_order=False, 
    width=None, height=None, rgb=False, title=None, 
    fontFamily="Helvetica-Narrow", keepPdfs=False, fontStyle=None,
    font_path="",
    min_font_size=40, max_font_size=250, max_words=500,
    big_mask=False,
    background_color="#FFFFFF",
    wordcloud_algorithm='ibm'):
    """given a list of words and a list of their frequencies, builds a wordle"""

    PERMA_path = os.path.dirname(os.path.realpath(__file__))

    if font_path == "":
        font_path = PERMA_path + "/meloche_bd.ttf"
    
    if wordcloud_algorithm == 'old': #old wordcloud function
        ro.r.library('Cairo')       #Optional, only used for old wordcloud module                                                                                                                              
        ro.r.library('wordcloud')   #Optional, only used for old wordcloud module 
        GRDEVICES = importr('grDevices')
        brewer = importr('RColorBrewer')
        
        assert(len(word_list) == len(freq_list))

        if width is None:
            width = 1280
        if height is None:
            height = 800

        size = len(word_list)
        ro_words = ro.StrVector(word_list)
        ro_freqs = ro.FloatVector(freq_list)
        ro_scale = ro.FloatVector([8, 1])

        ro_colors = None
        ordered_colors = False
        if color_list:
            if rgb:
                ro_colors = _makeColorTuplesFromRgbTuples(color_list)
            else:
                ro_colors = ro.StrVector(color_list)
                if not random_colors: ordered_colors = True

        else:
            ro_colors = _makeRandomColorTuples(size)

        if output_prefix: 
            # (if Cairo gives an error you need to install it locally through R: install.packages("Cairo"))
            if fontStyle:
                fontFamily = fontFamily + ':' + fontStyle
            ro.r.CairoFonts(regular=fontFamily)
            ro.r.Cairo(type="pdf", file="%s_wc.pdf"%output_prefix, width=11, height=8.5, units="in", onefile=True)
        ro.r.options(warn=2)
        while True:
            try:
                ro.r.wordcloud(ro_words, ro_freqs, rot_per=0, scale=ro_scale,max_words=25,random_order=random_order,colors=ro_colors, random_color=random_colors)
                if title:
                    ro.r.text(0, 1, labels=title, cex=0.8, pos=4)
                break
            except Exception as e:
                ro_scale[0] -= 1
                if ro_scale[0] <= ro_scale[1]:
                    raise Exception("Fatal Error: Either Words do not fit in wordle or error is due to R: [%s]"%e.message)
        ro.r.options(warn=0)
        if output_prefix:
            GRDEVICES.dev_off()
            # count number of pdf pages
            pdfFile = output_prefix + "_wc.pdf"
            nPages = -1
            with open(pdfFile, 'rb') as rf:
                for line in rf.readlines():
                    if "/Count " in line:
                        nPages = int( re.search("/Count \d*", line).group()[7:] )
                        # print pdfFile, 'has', nPages, 'pages'
            # convert the last page to png
            pngFile = output_prefix + "_wc.png"
            try:
                check_call(["convert", "-quality", "100", pdfFile + "[%d]"%(nPages-1), pngFile])
            except (OSError, CalledProcessError) as e:
                print 'ERROR:', e
            if not keepPdfs:
                check_call(["rm", pdfFile])

    elif wordcloud_algorithm == 'amueller':   #new wordcloud function
        f, pathname, desc = imp.find_module('wordcloud', sys.path[1:])
        wc = imp.load_module('wc', f, pathname, desc)
        if f is not None:
            f.close()
        #explicitly import amueller's wordcloud library
        #for all intents that block above is equivalent to: import wordcloud as wc
        #this is needed as both the local module and amueller's package are both called 'wordcloud'

        assert(len(word_list) == len(freq_list))
        color_string_list = None
        if width is None:
            width = 1280
        if height is None:
            height = 800

        print('Generating wordclouds using amueller\'s python package...')

        word_freq_tup = zip(word_list, freq_list)
        #list of tups of (word, freq)

        word_freq_tup = sorted(word_freq_tup, key=lambda tup: -tup[1]) #sort by frequency


        if color_list:
            #color list is list of colors to be applied to words with least to most freq
            assert(len(color_list) == len(word_list))
            if rgb:
                color_string_list = map(lambda x: "rgb(%d,%d,%d)" % (x[0], x[1], x[2]), color_list)
                #print color_string_list
            else:
                color_string_list = color_list #assume that the color list already has usable color strings
                if not random_colors: ordered_colors = True
        
        else:
            color_string_list = _makeRandomColorTuples(len(word_list))
            #create list of random color tuples, one for each word

        sorted_word_list, sorted_freq_list = zip(*word_freq_tup)


        
        color_dict = {sorted_word_list[i]: color_string_list[i] for i in range(0, len(sorted_word_list))} 
        #create dictionary where each word is mapped to a rgb value

        def color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
            try:
                return color_dict[word]
            except ValueError:
                return 'black'

        img_dir = os.path.dirname(os.path.abspath(__file__))
        if (big_mask is True or len(word_list) > 75): #arbitrary number for choosing mask
            img = Image.open(os.path.join(img_dir, 'oval_big_mask.png'))
        else:
            img = Image.open(os.path.join(img_dir, 'oval_mask.png'))

        img_array = array(img) #nd-array

        cloud = wc.WordCloud(font_path=font_path,
                    width=width,
                    height=height,
                    max_font_size=max_font_size,
                    min_font_size=min_font_size,
                    max_words=max_words,
                    color_func=color_func,
                    prefer_horizontal=1.0,
                    background_color=background_color,
                    mask=img_array).generate_from_frequencies(word_freq_tup)

        if output_prefix: 
            #TODO: pdf output?
            pngFile = output_prefix + "_wc.png"
            cloud.to_file(pngFile)
            print('Wordcloud created at: %s' % pngFile)
        else:
            warn('No filename specified. Wordcloud not created.')
            #Hey buddy. You didn't specify the filename.
    
    elif wordcloud_algorithm == 'ibm':
        if width is None:
            width = 1000
        if height is None:
            height = 2000

        if background_color[0] == "#": #TODO: do an actual HEX check
            background_color = background_color[1:]

        config_loc = PERMA_path + "/wc_config.txt";
        #if not os.path.isfile(config_loc):
        with open(config_loc, "w") as f:
            f.write('font: ' + font_path + '\n')
            f.write("format: tab\n" +
                    "inputencoding: UTF-8\n" +
                    "firstline: data\n" +
                    "wordcolumn: 1\n" +
                    "weightcolumn: 2\n" +
                    "colorcolumn: 3\n" +
                    "background: " + background_color + "\n" +
                    "placement: HorizontalCenterLine\n" +
                    "shape: BLOBBY\n" +
                    "orientation: HORIZONTAL")

        if not output_prefix:
            warn('No filename specified. Filename specified as \'wc\'.')
            output_prefix = 'wc'

        words_loc = os.path.normcase(output_prefix)
        words_loc = words_loc + '.txt'

        # if (output_prefix[:2] == './' or output_prefix[:1] == '/' ):
        #     words_loc = output_prefix 
        # else: #assuming relative path
        #     words_loc = "./" + output_prefix

        # if (output_prefix[-1:] == '/'):
        #     words_loc = words_loc[:-1] + '.txt'
        # else:
        #     words_loc = words_loc + '.txt'   


        with open(words_loc, 'w') as f:
            word_freq_tup = zip(word_list, freq_list)
            word_freq_tup = sorted(word_freq_tup, key=lambda tup: -tup[1]) #sort by frequency

            if color_list:
                #color list is list of colors to be applied to words with least to most freq
                assert(len(color_list) == len(word_list))
                if rgb:
                    color_string_list = map(lambda x: "rgb(%d,%d,%d)" % (x[0], x[1], x[2]), color_list)
                    #print color_string_list
                else:
                    color_string_list = color_list #assume that the color list already has usable color strings
                    if not random_colors: ordered_colors = True
            
            else:
                color_string_list = _makeRandomColorTuples(len(word_list))
                #create list of random color tuples, one for each word

            sorted_word_list, sorted_freq_list = zip(*word_freq_tup)
            word_freq_color_tup = zip(sorted_word_list, sorted_freq_list, color_string_list)


            pngFile = output_prefix + '.png'

            for tup in word_freq_color_tup:
                try:
                    word_row = tup[0].encode("utf-8") + '\t' + str(tup[1]) + '\t' + str(tup[2]) + '\n'
                    f.write(word_row)
                except Exception,e:
                    print e
                    print "Line contains unprintable unicode, skipped: ",
                    print tup
                #str is necessary in case the words are numbers

        command = ['java','-jar', PERMA_path + '/ibm-word-cloud.jar', '-c', 
                    config_loc, '-w', str(width), '-h', str(height), '-i', 
                    words_loc, '-o', pngFile]
        print 'Generating wordcloud at ' + pngFile

        #print ' '.join(command)
        with open(os.devnull, 'w') as fnull: #for suppressing outputs of the command
            subprocess.call(command, stdout=fnull, stderr=subprocess.STDOUT)

    else:
        print "Wordcloud algorithm not recognized."
        print "Change line 122 of PERMA/code/ml/wwbp/wordcloud.py to valid algorithm."

def _tagcloudToWordcloud(filename='', directory=os.getcwd()):
    f = open(os.path.join(directory, filename), 'rb')
    lastline = ''
    last_starline = False
    bracketline = False
    getwords = False
    tagTriples = []
    for line in f.readlines():
        line = line.strip()
        if getwords:
            if not line:
                getwords = False
                topicId = extract(lastline, 'Topic Id:',',').strip()
                # tags = map(lambda x:(x[0].upper(), int(x[1]), rgb(x[2])), tagTriples)
                tags = map(lambda x:(x[0].lower(), int(x[1]), rgb(x[2])), tagTriples)
                ngrams, freqs, colors = zip(*tags)
                lastline = ''
                tagTriples = []
#                print topicId, ngrams, freqs, colors
                wordcloud(ngrams, freqs, os.path.join(directory, topicId),colors,rgb=True)
                break
            else:
                tagTriples.append(explode(line))

        if len(line) > 0 and line[0] == '[' and line[-1] == ']':
            bracketline = True
            lastline = line
            if not last_starline:
                getwords = True
            bracketline = False
        if len(line) > 1 and line[0:2] == '**' and line[-2:]:
            last_starline = True
        else:
            last_starline = False

    f.close()

def processTopicLine(line):
    splitLine = line.split(' ')
    return 'tid-'+splitLine[2][0:-1], 'tR-'+splitLine[4][0:-1][0:5], '' #'tFq'+splitLine[6][0:-1]

def duplicateFilterLineIntoInformativeString(line):
    splitLine = line.split(' ')
    return 'df-%s-of-%s'%(splitLine[1], splitLine[5])

def coerceToValidFileName(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)

def tagcloudToWordcloud(filename='', directory='', withTitle=False, fontFamily="Helvetica-Narrow", fontStyle=None, toFolders=False, useTopicWords=True):
    processedFiles = {}

    if not directory:
        directory = os.path.dirname(filename)        
        filename = os.path.basename(filename)

    fileFolders = {}
    filename = filename + '.txt'
    basefilename = filename[:-4]
    # basefolder = os.path.join(directory, basefilename + '.output-dir')
    basefolder = os.path.join(directory, basefilename + '_wordclouds')
    
    with open(os.path.join(directory, filename), 'rb') as f:
        lastline = ''; tagTriples = [];
        getwords = False; posneg=''; isNewTagcloud = False;
        outcome = ''; topicId = None; topicR = None; topicFreq = None;
        topicDup = '';

        for line in f.readlines():
            line = line.strip()

            if getwords:
                if not line and len(tagTriples)==0:
                    getwords = False
                elif line and len(line) >= 19 and line[0:19]=='rList has less than':
                    getwords = False
                elif line and len(line) >= 18 and line[0:18]=='Error_in_function_':
                    line = None
                elif line and len(line) >= 28 and line[0:28]=='rList has less than no items':
                    getwords = False
                elif '(converted_from_warning)' in line:
                    line = None
                elif line:
                    tagTriples.append(explode(line))

                else:
                    tags = map(lambda x:(x[0].lower(), int(x[1]), x[2]), tagTriples)
                    newFilename = None
                    fileEnding = ''
                    fileEnding += '.' + outcome
                    fileEnding += '.' + posneg
                    if topicR: fileEnding += '.' + topicR
                    if topicId: fileEnding += '.' + topicId
                    if useTopicWords:
                        ngrams, freqs, colors = zip(*tags)
                        fileEnding += '.'.join(ngrams[0:6])
                    if topicFreq: fileEnding += '.' + topicFreq
                    if topicDup: fileEnding += '.' + topicDup
                    fileEnding += '.txt'
                    newFilename = filename.replace('.txt', fileEnding) if '.txt' in filename else filename + fileEnding
                    newFilename = coerceToValidFileName(newFilename)
                    processedFiles[newFilename[0:-4]] = tags
                    topicR = '' if not topicR else topicR
                    topicId = '' if not topicId else topicId
                    topicDup = '' if not topicDup else topicDup
                    topicFreq = '' if not topicFreq else topicFreq

                    if toFolders:
                        ngrams, freqs, colors = zip(*tags)
                        ngramString = '.'.join(ngrams[0:min(6, len(ngrams))])
                        ngramString = '' ## LAD delete this line to add ngrams / entries back into the file
                        fileFolders[newFilename[0:-4]] = (outcome, '.'.join([posneg, topicR, topicId, ngramString, topicDup, topicFreq]))

                    getwords = False; tagTriples = [];
                    topicDup = ''; topicId = None; topicR = None; topicFreq = None;


            if line and len(line) >= 10 and line[0:10] == '[Topic Id:':
                topicId, topicR, topicFreq = processTopicLine(line)
                getwords = True
            if line and len(line)>=6 and line[0:3]=='**[' and line[-3:]==']**':
                topicDup = duplicateFilterLineIntoInformativeString(line)
            if line and len(line)==12 and line == '------------':
                posneg = 'pos'; getwords = True;
            if line and len(line)==13 and line == '-------------':
                posneg = 'neg'; getwords = True;
            if line and len(line)==27 and line[0:27] == '---------------------------':
                outcome = lastline[19:];
            if line and len(line)==32  and line[0:32] == '--------------------------------':
                outcome = lastline[25:]
            lastline = line

        if getwords:
            tags = map(lambda x:(x[0], int(x[1]), x[2]), tagTriples)
            if tags:
                # ngrams, freqs, colors = zip(*tags)
                newFilename = None
                fileEnding = ''
                fileEnding += '.' + outcome
                fileEnding += '.' + posneg
                if topicR: fileEnding += '.' + topicR
                if topicId: fileEnding += '.' + topicId
                if useTopicWords:
                    ngrams, freqs, colors = zip(*tags)
                    fileEnding += '.'.join(ngrams[0:6])
                if topicFreq: fileEnding += '.' + topicFreq
                if topicDup: fileEnding += '.' + topicDup
                fileEnding += '.txt'
                newFilename = filename.replace('.txt', fileEnding) if '.txt' in filename else filename + fileEnding
                processedFiles[newFilename[0:-4]] = tags
                topicR = '' if not topicR else topicR
                topicId = '' if not topicId else topicId
                topicDup = '' if not topicDup else topicDup
                topicFreq = '' if not topicFreq else topicFreq


                newFilename = coerceToValidFileName(newFilename)
                if toFolders:
                    ngrams, freqs, colors = zip(*tags)
                    ngramString = '.'.join(ngrams[0:min(6, len(ngrams))])
                    ngramString = '' ## LAD delete this line to add ngrams / entries back into the file
                    suffixList = [posneg, topicR, topicId, ngramString, topicDup, topicFreq]
                    fileFolders[newFilename[0:-4]] = (outcome, '.'.join(suffixList))
    
    # from pprint import pprint as pp
    ii = 0

    # import pdb
    # pdb.set_trace()

    if toFolders:
        try:
            os.mkdir(basefolder)
        except OSError:
            pass

    for filename, tags in processedFiles.iteritems():
        ngrams, freqs, colors = zip(*tags)
        colors = map(lambda x:'#'+x, colors)
        filename = coerceToValidFileName(filename)
        title = filename if withTitle else None
        output_file = None
        if toFolders:
            outcome, endname = fileFolders[filename]
            try:
                os.mkdir(os.path.join(basefolder, outcome))
            except OSError:
                pass
            output_file = os.path.join(basefolder, outcome, endname)
        else:
            output_file = os.path.join(directory, filename)
        try:
            wordcloud(ngrams, freqs,output_file, colors, rgb=False, title=title, fontFamily=fontFamily, fontStyle=fontStyle)
        except Exception, e:
            print 'WARNING: ERROR happened for file: %s'%(output_file,)
            print e
        
        # ii += 1
        # if ii > 2:
        #     break

    return processedFiles
    



if __name__=='__main__':

    print 'wordcloud sleeps peacefully in the shade, its work is happily done.'
