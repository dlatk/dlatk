#!/usr/bin/python
###########################################################
## featureWorker.py
##
## Interface Module to extract features and create tables holding the features
##
## andy schwartz    - fall 2011 - hansens@sas.upenn.edu
## luke dziurzynski - fall 2011 - lukaszdz@sas.upenn.edu
##
## TODO:
## -handle that mysql is not using mixed case (should be lowercase all features?)
## -convert argument parser to be its own object that can be inherited
## -with zeros should consider that the data may have been transformed...?

__authors__ = "H. Andrew Schwartz, Lukasz Dziurzynski, Megha Agrawal, Sneha Jha @ University of Pennsylvania"
__copyright__ = "Copyright 2013"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "0.3"
__maintainer__ = "H. Andrew Schwartz"
__email__ = "hansens@sas.upenn.edu"

import os, getpass
import sys
import pdb
import argparse
import time
from pprint import pprint
from numpy import isnan, sqrt, log2
from FeatureWorker import DDLA
from ConfigParser import SafeConfigParser

#wwbp
try:
    from FeatureWorker.lib.StanfordSegmenter import StanfordSegmenter
except ImportError:
    print 'warning: StanfordSegmenter not found.'
from FeatureWorker.lib import wordcloud
from FeatureWorker.semanticsExtractor import SemanticsExtractor
import FeatureWorker.featureWorker as featureWorker
from FeatureWorker.regressionPredictor import RegressionPredictor, CombinedRegressionPredictor, ClassifyToRegressionPredictor
from FeatureWorker.classifyPredictor import ClassifyPredictor
from FeatureWorker.clustering import DimensionReducer, CCA
from FeatureWorker.mediation import MediationAnalysis

# INTERFACE_PATH = '../../data/lexicons/interface' #os.path.dirname(__file__)+'/../../data/lexicons/interface/'
INTERFACE_PATH = os.path.dirname(os.path.abspath(featureWorker.__file__))+'/LexicaInterface'
sys.path.append(INTERFACE_PATH)
try:
    import lexInterface
except ImportError:
    print 'warning: interface module not imported..'

from FeatureWorker.featureWorker import FeatureWorker
from FeatureWorker.featureExtractor import FeatureExtractor
from FeatureWorker.outcomeGetter import OutcomeGetter
from FeatureWorker.outcomeAnalyzer import OutcomeAnalyzer
from FeatureWorker.featureGetter import FeatureGetter
from FeatureWorker.featureRefiner import FeatureRefiner
#defaults:

HOST = ''
USER = getpass.getuser()

MAX_ATTEMPTS = 10 #max number of times to try a query before exiting
PROGRESS_AFTER_ROWS = 5000 #the number of rows to process between each progress updated
FEATURE_TABLE_PREFIX = 'feats_'
MYSQL_ERROR_SLEEP = 4 #number of seconds to wait before trying a query again (incase there was a server restart
MYSQL_BATCH_INSERT_SIZE = 10000 # how many rows are inserted into mysql at a time
VARCHAR_WORD_LENGTH = 24 #length to allocate var chars per words
LOWERCASE_ONLY = 1#if the db is case insensitive, set to 1
MAX_TO_DISABLE_KEYS = 300000 #number of groups * n must be less than this to disable keys
MAX_SQL_PRINT_CHARS = 256

##Corpus Settings:
DEF_CORPDB = 'fb20'
DEF_CORPTABLE = 'messages_en'
DEF_CORREL_FIELD = 'user_id'
DEF_MESSAGE_FIELD = 'message'
DEF_MESSAGEID_FIELD = 'message_id'
#DEF_MESSAGEID_FIELD = 'id'
DEF_LEXTABLE = 'wn_O'
DEF_DATE_FIELD = 'updated_time'

##Outcome settings
DEF_OUTCOME_TABLE = 'masterstats_andy_maxAge64'
DEF_OUTCOME_FIELDS = []
DEF_OUTCOME_CONTROLS = []
DEF_GROUP_FREQ_THRESHOLD = int(1000) #min. number of total feature values that the group has, to use it
DEF_OUTPUTDIR = '/data/ml/fb20'
DEF_SHOW_FEAT_FREQS = True
DEF_MAX_TC_WORDS = 100
DEF_TC_FILTER = True

##Feature Settings:
DEF_N = int(1);
DEF_FEAT_NAMES = ['honor']
DEF_MIN_FREQ = int(1); #min frequency per group to keep (don't advise above 1)
DEF_P_OCC = float(.01);#percentage of groups a feature must appear in, to keep it
DEF_PMI = 3.0;
DEF_MIN_FEAT_SUM = 0 #minimum sum of feature total to keep
DEF_STANFORD_POS_MODEL = '/home/hansens/Tools/StanfordTagger/stanford-postagger-2012-01-06/models/english-bidirectional-distsim.tagger'
DEF_LEXICON_DB = 'permaLexicon'
DEF_FEAT_TABLE = 'feat$1gram$messages_en$user_id$16to16$0_01'
DEF_COLLOCTABLE = 'test_collocs'
DEF_COLUMN_COLLOC = "feat"
DEF_COLUMN_PMI_FILTER = "pmi_filter_val"
DEF_P = 0.05 # p value for printing tagclouds

##Mediation Settings:
DEF_MEDIATION_BOOTSTRAP = 1000
DEF_MEDIATION_P = 0.05
DEF_OUTCOME_PATH_STARTS = []
DEF_OUTCOME_MEDIATORS = []

DEF_TOPIC_LEX_METHOD = 'csv_lik'

##Prediction Settings:
DEF_MODEL = 'ridgecv'
DEF_CLASS_MODEL = 'svc'
DEF_COMB_MODELS = ['ridgecv']
DEF_FOLDS = 5

##Meta settings
DEF_INIT_FILE = 'initFile.txt'

def _warn(string):
    print >>sys.stderr, string

def _getReportingInt(reporting_percent, iterable_length):
    return max(1, floor(reporting_percent * iterable_length))

def _report(object_description, ii, reporting_int, iterable_length):
    #print ii, reporting_int
    if ii % reporting_int == 0:
        _warn("%d out of %d %s processed; %2.2f complete"%(ii, iterable_length, object_description, float(ii)/iterable_length))

def getInitVar(variable, parser, default, varList=False):
    if parser:
        if varList:
            return [o.strip() for o in parser.get('constants',variable).split(",")] if parser.has_option('constants',variable) else default
        else:
            return parser.get('constants',variable) if parser.has_option('constants',variable) else default
    else:
        return default

#################################################################
### Main / Command-Line Processing:
##
#
def main(fn_args = None):
    '''
    :param fn_args: string - ex "-d testing -t msgs -c user_id --add_ngrams -n 1 "
    '''
    start_time = time.time()

    ##Argument Parser:
    parser = argparse.ArgumentParser(
        description='Extract and Manage Language Feature Data.', prefix_chars='-+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Meta variables
    group = parser.add_argument_group('Meta Variables', '')
    group.add_argument('--to_file', dest='toinitfile', nargs='?', const=DEF_INIT_FILE, default=None,
                      help='write flag values to text file')
    group.add_argument('--from_file', type=str, dest='frominitfile', default='',
                       help='reads flag values from file')

    args, remaining_argv = parser.parse_known_args()

    if args.frominitfile:
        conf_parser = SafeConfigParser()
        conf_parser.read(args.frominitfile)
    else:
        conf_parser = None

    group = parser.add_argument_group('Corpus Variables', 'Defining the data from which features are extracted.')

    group.add_argument('-d', '--corpdb', metavar='DB', dest='corpdb', default=getInitVar('corpdb', conf_parser, DEF_CORPDB),
                        help='Corpus Database Name.')
    group.add_argument('-t', '--corptable', metavar='TABLE', dest='corptable', default=getInitVar('corptable', conf_parser, DEF_CORPTABLE),
                        help='Corpus Table.')
    group.add_argument('-c', '--correl_field', metavar='FIELD', dest='correl_field', default=getInitVar('correl_field', conf_parser, DEF_CORREL_FIELD),
                        help='Correlation Field (AKA Group Field): The field which features are aggregated over.')
    group.add_argument('-H', '--host', metavar='HOST', dest='mysql_host', default=getInitVar('mysql_host', conf_parser, HOST),
                       help='Host that the mysql server runs on (default: %s)' % HOST)
    group.add_argument('--message_field', metavar='FIELD', dest='message_field', default=getInitVar('message_field', conf_parser, DEF_MESSAGE_FIELD),
                        help='The field where the text to be analyzed is located.')
    group.add_argument('--messageid_field', metavar='FIELD', dest='messageid_field', default=getInitVar('messageid_field', conf_parser, DEF_MESSAGEID_FIELD),
                        help='The unique identifier for the message.')
    group.add_argument('--date_field', metavar='FIELD', dest='date_field', default=getInitVar('date_field', conf_parser, DEF_DATE_FIELD),
                        help='Date a message was sent (if avail, for timex processing).')
    group.add_argument('--lexicondb', metavar='DB', dest='lexicondb', default=getInitVar('lexicondb', conf_parser, DEF_LEXICON_DB),
                        help='The database which stores all lexicons.')

    group = parser.add_argument_group('Feature Variables', 'Use of these is dependent on the action.')
    group.add_argument('-f', '--feat_table', metavar='TABLE', dest='feattable', type=str, nargs='+', default=getInitVar('feattable', conf_parser, None, varList=True),
                       help='Table containing feature information to work with')
    group.add_argument('-n', '--set_n', metavar='N', dest='n', type=int, nargs='+', default=[DEF_N],
                       help='The n value used for n-grams or co-occurence features')
    group.add_argument('--no_metafeats', action='store_false', dest='metafeats', default=True,
                       help='indicate not to extract meta features (word, message length) with ngrams')
    group.add_argument('-l', '--lex_table', metavar='TABLE', dest='lextable', default=getInitVar('lextable', conf_parser, ''),
                       help='Lexicon Table Name: used for extracting category features from 1grams'+
                       '(or use --word_table to extract from other than 1gram)')
    group.add_argument('--word_table', metavar='WORDTABLE', dest='wordTable', default=getInitVar('wordTable', conf_parser, None),
                       help='Table that contains the list of words to give for lex extraction/group_freq_thresh')
    group.add_argument('--colloc_table', metavar='TABLE', dest='colloc_table', default=DEF_COLLOCTABLE,
                        help='Table that holds a list of collocations to be used as features.')
    group.add_argument('--colloc_column', metavar='COLUMN', dest='colloc_column', default=DEF_COLUMN_COLLOC,
                        help='Column giving collocations to be used as features.')
    group.add_argument('--feature_type_name', metavar='STRING', dest='feature_type_name',
                        help='Customize the name of output features.')
    group.add_argument('--gzip_csv', metavar='filename', dest='gzipcsv', default='',
                       help='gz-csv filename used for extracting ngrams')
    group.add_argument('--categories', type=str, metavar='CATEGORY(IES)', dest='categories', nargs='+', default='',
                       help='Specify particular categories.')
    group.add_argument('--feat_blacklist', type=str, metavar='FEAT(S)', dest='feat_blacklist', nargs='+', default='',
                       help='Features to ban when correlating with outcomes.')
    group.add_argument('--feat_whitelist', type=str, metavar='FEAT(S)', dest='feat_whitelist', nargs='+', default='',
                       help='Only permit these features when correlating with outcomes (specify feature names or if feature table then read distinct features).')
    group.add_argument('--sqrt', action='store_const', dest='valuefunc', const=lambda d: sqrt(d),
                       help='square-roots normalized group_norm freq information.')
    group.add_argument('--log', action='store_const', dest='valuefunc', const=lambda d: log2(d+1),
                       help='logs the normalized group_norm freq information.')
    group.add_argument('--anscombe', action='store_const', dest='valuefunc', const=lambda d: 2*sqrt(d+3/float(8)),
                       help='anscombe transforms normalized group_norm freq information.')
    group.add_argument('--boolean', action='store_const', dest='valuefunc', const=lambda d: float(1.0),
                       help='boolean transforms normalized group_norm freq information (1 if true).')
    group.add_argument('--lex_sqrt', action='store_const', dest='lexvaluefunc', const=lambda d: sqrt(d),
                       help='square-roots normalized group_norm lexicon freq information.')
    group.add_argument('--lex_log', action='store_const', dest='lexvaluefunc', const=lambda d: log2(d+1),
                       help='logs the normalized group_norm lexicon freq information.')
    group.add_argument('--lex_anscombe', action='store_const', dest='lexvaluefunc', const=lambda d: 2*sqrt(d+3/float(8)),
                       help='anscombe transforms normalized group_norm lexicon freq information.')
    group.add_argument('--lex_boolean', action='store_const', dest='lexvaluefunc', const=lambda d: float(1.0),
                       help='boolean transforms normalized group_norm freq information (1 if true).')
    # group.add_argument('--gender', type=str, dest='gender', default=None,
    #                    help='gender input (used by flexibin)')
    # group.add_argument('--gender_attack', action='store_true', dest='genderattack', default=False,
    #                    help='indicates dual gender table input (used by flexibin)')
    # group.add_argument('--is_binned', action='store_true', dest='isbinned', default=False,
    #                    help='indicates we are using a group_id binned feature table')
    # group.add_argument('--lex_thresh', type=float, dest='lexthresh', default=None,
    #                   help='threshold: retains all lda topics with scores x such that x >= threshold*maximium topic score')
#removed because we should use filter instead
#    group.add_argument('--set_min_freq', metavar='N', dest='min_freq', type=int, default=DEF_MIN_FREQ,
#                       help='The minimum frequency of a feature per group to store it (strongly advise against > 1).')
    group.add_argument('--set_p_occ', metavar='P', dest='pocc', type=float, default=DEF_P_OCC,
                       help='The probability of occurence of either a feature or group (altnernatively if > 1, then limits to top p_occ features instead).')
    group.add_argument('--set_pmi_threshold', metavar='PMI', dest='pmi', type=float, default=DEF_PMI,
                       help='The threshold for the feat_colloc_filter.')
    group.add_argument('--set_min_feat_sum', metavar='N', dest='minfeatsum', type=int, default=DEF_MIN_FEAT_SUM,
                       help='The minimum a feature must occur across all groups, to be kept.')
    group.add_argument('--topic_file', type=str, dest='topicfile', default='',
                       help='Name of topic file to use to build the topic lexicon.')
    group.add_argument('--num_topic_words', type=int, dest='numtopicwords', default=5,
                       help='Number of topic words to use as labels.')
    group.add_argument('--topic_lexicon', type=str, dest='topiclexicon', default='',
                       help='this is the (topic) lexicon name specified as part of --make_feat_labelmap_lex and --add_topiclex_from_topicfile')
    group.add_argument('--topic_list', type=str, dest='topiclist', default='', nargs='+',
                       help='this is the list of topics to group together in a plot for --feat_flexibin')
    group.add_argument('--topic_lex_method', type=str, dest='topiclexmethod', default=DEF_TOPIC_LEX_METHOD,
                       help='must be one of: "csv_lik", "standard"')
    group.add_argument('--weighted_lexicon', action='store_true', dest='weightedlexicon', default=False,
                       help='use with Extraction Action add_lex_table to make weighted lexicon features')
    group.add_argument('--num_bins', type=int, dest='numbins', default=None,
                       help='number of bins (feature refiner).')
    group.add_argument('--flexiplot_file', type=str, dest='flexiplotfile', default='',
                       help='use with Plot Action --feat_flexibin to specify a file to read for plotting')
    group.add_argument('--group_id_range', type=float, dest='groupidrange', nargs=2, 
                       help='range of group id\'s to include in binning.')
    group.add_argument('--mask_table', type=str, metavar='TABLE', dest='masktable', default=None,
                       help='Table containing which groups run in various bins (for ttest).')



    group = parser.add_argument_group('Outcome Variables', '')
    group.add_argument('--outcome_table', type=str, metavar='TABLE', dest='outcometable', default=getInitVar('outcometable', conf_parser, DEF_OUTCOME_TABLE),
                       help='Table holding outcomes (make sure correl_field type matches corpus\').')
    group.add_argument('--outcome_fields', '--outcomes',  type=str, metavar='FIELD(S)', dest='outcomefields', nargs='+', default=getInitVar('outcomefields', conf_parser, DEF_OUTCOME_FIELDS, varList=True),
                       help='Fields to compare with.')
    group.add_argument('--outcome_controls', '--controls', type=str, metavar='FIELD(S)', dest='outcomecontrols', nargs='+', default=getInitVar('outcomecontrols', conf_parser, DEF_OUTCOME_CONTROLS, varList=True),
                       help='Fields in outcome table to use as controls for correlation(regression).')
    group.add_argument('--outcome_interaction', '--interaction', type=str, metavar='TERM(S)', dest='outcomeinteraction', nargs='+', default=getInitVar('outcomeinteraction', conf_parser, DEF_OUTCOME_CONTROLS, varList=True),
                       help='Fields in outcome table to use as controls and interaction terms for correlation(regression).')
    group.add_argument('--feat_names', type=str, metavar='FIELD(S)', dest='featnames', nargs='+', default=getInitVar('featnames', conf_parser, DEF_FEAT_NAMES, varList=True),
                       help='Limit outputs to the given set of features.')
    group.add_argument("--group_freq_thresh", type=int, metavar='N', dest="groupfreqthresh", default=getInitVar('groupfreqthresh', conf_parser, int(DEF_GROUP_FREQ_THRESHOLD)),
                       help="minimum WORD frequency per correl_field to include correl_field in results")
#    group.add_argument('--output_dir', type=str, metavar='DIR', dest='outputdir', default=DEF_OUTPUTDIR,
#                       help='Destination for html or plot images (depricated, use output_name).')
    group.add_argument('--output_name', '--output', type=str, dest='outputname', default=getInitVar('outputname', conf_parser, ''),
                       help='overrides the default filename for output')
    group.add_argument('--max_tagcloud_words', type=int, metavar='N', dest='maxtcwords', default=DEF_MAX_TC_WORDS,
                       help='Max words to appear in a tagcloud')
    group.add_argument('--show_feat_freqs', action='store_true', dest='showfeatfreqs', default=DEF_SHOW_FEAT_FREQS,)
    group.add_argument('--not_show_feat_freqs', action='store_false', dest='showfeatfreqs', default=DEF_SHOW_FEAT_FREQS,
                       help='show / dont show feature frequencies in output.')
    group.add_argument('--tagcloud_filter', action='store_true', dest='tcfilter', default=DEF_TC_FILTER,)
    group.add_argument('--no_tagcloud_filter', action='store_false', dest='tcfilter', default=DEF_TC_FILTER,
                       help='filter / dont filter tag clouds for duplicate info in phrases.')
    group.add_argument('--feat_labelmap_table', type=str, dest='featlabelmaptable', default=getInitVar('featlabelmaptable', conf_parser, ''),
                       help='specifies an lda mapping tablename to be used for LDA topic mapping')
    group.add_argument('--feat_labelmap_lex', type=str, dest='featlabelmaplex', default=getInitVar('featlabelmaplex', conf_parser, ''),
                       help='specifies a lexicon tablename to be used for the LDA topic mapping')
    group.add_argument('--bracket_labels', action='store_true', dest='bracketlabels', default='',
                       help='use with: feat_labelmap_lex... if used, the labelmap features will be contained within brackets')
    group.add_argument('--comparative_tagcloud', action='store_true', dest='compTagcloud', default=False,
                       help='used with --sample1 and --sample2, this option uses IDP to compare feature usage')
    group.add_argument('--sample1', type=str, nargs='+', dest="compTCsample1", default=[],
                       help='first sample of group to use in comparison [use with --comparative_tagcloud]'+
                       "(use * to mean all groups in featuretable)")
    group.add_argument('--sample2', type=str, nargs='+', dest="compTCsample2", default=[],
                       help='second sample of group to use in comparison [use with --comparative_tagcloud]'+
                       "(use * to mean all groups in featuretable)")
    group.add_argument('--csv', action='store_true', dest='csv', 
                       help='generate csv correl matrix output as well')
    group.add_argument('--pickle', action='store_true', dest='pickle', 
                       help='generate pickle of the correl matrix output as well')
    group.add_argument('--sort', action='store_true', dest='sort', 
                       help='add sorted output for correl matrix')
    group.add_argument('--whitelist', action='store_true', dest='whitelist', default=False,
                       help='Uses feat_whitelist or --lex_table and --categories.')
    group.add_argument('--blacklist', action='store_true', dest='blacklist', default=False,
                       help='Uses feat_blacklist or --lex_table and --categories.')
    group.add_argument('--spearman', action='store_true', dest='spearman', 
                       help='Use Spearman R instead of Pearson.')
    group.add_argument('--logistic_reg', action='store_true', dest='logisticReg', default=False,
                       help='Use logistic regression instead of linear regression. This is better for binary outcomes.')
    group.add_argument('--IDP', '--idp', action='store_true', dest='IDP', default=False,
                       help='Use IDP instead of linear regression/correlation [only works with binary outcome values]') 
    group.add_argument('--AUC', '--auc', action='store_true', dest='auc', default=False,
                       help='Use AUC instead of linear regression/correlation [only works with binary outcome values]') 
    group.add_argument('--zScoreGroup', action='store_true', dest='zScoreGroup', default=False,
                       help="Outputs a certain group's zScore for all feats, which group is determined by the boolean outcome value [MUST be boolean outcome]") 
    group.add_argument('--p_correction', metavar='METHOD', type=str, dest='p_correction_method', default=getInitVar('p_correction_method', conf_parser, ''),
                       help='Specify a p-value correction method: simes, holm, hochberg, hommel, bonferroni, BH, BY, fdr, none')
    group.add_argument('--no_bonferroni', action='store_false', dest='bonferroni', default=True,
                       help='Turn off bonferroni correction of p-values.')
    group.add_argument('--nvalue', type=bool, dest='nvalue', default=True,
                       help='Report n values.')
    group.add_argument('--freq', type=bool, dest='freq', default=True,
                       help='Report freqs.')
    group.add_argument('--tagcloud_colorscheme', type=str, dest='tagcloudcolorscheme', default=getInitVar('tagcloudcolorscheme', conf_parser, 'multi'), 
                       help='specify a color scheme to use for tagcloud generation. Default: multi, also accepts red, blue, & red-random')
    group.add_argument('--interactions', action='store_true', dest='interactions', default=False,
                       help='Includes interaction terms in multiple regression.')
    group.add_argument('--bootstrapp', '--bootstrap', dest='bootstrapp', type=int, default = 0,
                       help="Bootstrap p-values (only works for AUCs for now) ")
    group.add_argument("--p_value", type=float, metavar='P', dest="maxP", default = getInitVar('maxP', conf_parser, float(DEF_P)),
                       help="Significance threshold for returning results. Default = 0.05.")

    group = parser.add_argument_group('Mediation Variables', '')
    group.add_argument('--mediation', action='store_true', dest='mediation', default=False,
                       help='Run mediation analysis.')
    group.add_argument('--mediation_bootstrap', '--mediation_boot', action='store_true', dest='mediationboot', default=False,
                       help='Run mediation analysis with bootstrapping. The parametric (non-bootstrapping) method is default.')
    group.add_argument("--mediation_boot_num", type=int, metavar='N', dest="mediationbootnum", default = int(DEF_MEDIATION_BOOTSTRAP),
                       help="The number of repetitions to run in bootstrapping with mediation analysis. Default = 1000.")
    group.add_argument('--outcome_pathstarts', '--path_starts', type=str, metavar='FIELD(S)', dest='outcomepathstarts', nargs='+', default=DEF_OUTCOME_PATH_STARTS,
                       help='Fields in outcome table to use as treatment in mediation analysis.')
    group.add_argument('--outcome_mediators', '--mediators', type=str, metavar='FIELD(S)', dest='outcomemediators', nargs='+', default=DEF_OUTCOME_MEDIATORS,
                       help='Fields in outcome table to use as mediators in mediation analysis.')
    group.add_argument('--feat_as_path_start', action='store_true', dest='feat_as_path_start', default=False,
                       help='Use path start variables located in a feature table. Used in mediation analysis.')
    group.add_argument('--feat_as_outcome', action='store_true', dest='feat_as_outcome', default=False,
                       help='Use outcome variables located in a feature table. Used in mediation analysis.')
    group.add_argument('--feat_as_control', action='store_true', dest='feat_as_control', default=False,
                       help='Use control variables located in a feature table. Used in mediation analysis.')
    group.add_argument('--no_features', action='store_true', dest='no_features', default=False,
                       help='All mediation analysis variables found corptable. No feature table needed.')
    group.add_argument('--mediation_csv', action='store_true', dest='mediationcsv', default=False,
                       help='Print results to a CSV. Default file name is mediation.csv. Use --output_name to specify file name.')
    group.add_argument('--mediation_mysql', action='store_true', dest='mediationmysql', default=False,
                       help='Store results in MySQL database. Database is specified by -d and table name is specified by -output_name')
    group.add_argument("--mediation_sig_thresh", type=float, metavar='P', dest="mediationsigthresh", default = float(DEF_MEDIATION_P),
                       help="Significance threshold for returning results. Default = 0.05.")
    group.add_argument('--mediation_no_summary', action='store_false', dest='mediationsummary', default=True,
                       help='Print results to a CSV. Default file name is mediation.csv. Use --output_name to specify file name.')



    group = parser.add_argument_group('Prediction Variables', '')
    group.add_argument('--model', type=str, metavar='name', dest='model', default=DEF_MODEL,
                       help='Model to use when predicting: svc, linear-svc, ridge, linear.')
    group.add_argument('--combined_models', type=str, nargs='+', metavar='name', dest='combmodels', default=DEF_COMB_MODELS,
                       help='Model to use when predicting: svc, linear-svc, ridge, linear.')
    group.add_argument('--sparse', action='store_true', dest='sparse', default=False,
                       help='use sparse representation for X when training / testing')
    group.add_argument('--folds', type=int, metavar='NUM', dest='folds', default=DEF_FOLDS,
                       help='Number of folds for functions that run n-fold cross-validation')
    group.add_argument('--picklefile', type=str, metavar='filename', dest='picklefile', default='',
                       help='Name of file to save or load pickle of model')
    group.add_argument('--all_controls_only', action='store_true', dest='allcontrolsonly', default=False,
                       help='Only uses all controls when prediction doing test_combo_regression')
    group.add_argument('--no_lang', action='store_true', dest='nolang', default=False,
                       help='Runs with language features excluded')
    group.add_argument('--control_combo_sizes', '--combo_sizes', type=int, metavar="index", nargs='+', dest='controlcombosizes', 
                       default=[], help='specify the sizes of control combos to use')
    group.add_argument('--residualized_controls', '--res_controls', action='store_true', dest='res_controls', default=False,
                       help='Finds residuals for controls and tries to predict beyond them (only for combo test)')
    group.add_argument('--prediction_csv', '--pred_csv', action='store_true', dest='pred_csv',
                       help='write yhats in a separate csv')
    group.add_argument('--weighted_eval', type=str, dest='weightedeval', default=None,
                       help='Column to weight the evaluation.')
    group.add_argument('--no_standardize', action='store_false', dest='standardize', default=True,
                       help='turn off standardizing variables before prediction')


    group = parser.add_argument_group('Standard Extraction Actions', '')
    group.add_argument('--add_ngrams', action='store_true', dest='addngrams',
                       help='add an n-gram feature table. (uses: n, can flag: sqrt), gzip_csv'
                       'can be used with or without --use_collocs')
    group.add_argument('--add_ngrams_from_tokenized', action='store_true', dest='addngramsfromtok',
                       help='add an n-gram feature table from a tokenized table. Table must be JSON list of tokens.'
                       '(uses: n, can flag: sqrt), gzip_csv.')
    group.add_argument('--use_collocs', action='store_true', dest='use_collocs',
                       help='when extracting ngram features, use a table of collocations to parse text into ngrams'
                            'by default does not include subcollocs, this can be changed with the --include_sub_collocs option ')
    group.add_argument('--include_sub_collocs', action='store_true', dest='include_sub_collocs',
                       help='count all sub n-grams of collocated n-grams'
                            'if "happy birthday" is designated as a collocation, when you see "happy birthday" in text'
                            'count it as an instance of "happy", "birthday", and "happy birthday"')
    group.add_argument('--colloc_pmi_thresh', metavar="PMI", dest='colloc_pmi_thresh', type=float, default=DEF_PMI,
                       help='The PMI threshold for which multigrams from the colloctable to conscider as valid collocs'
                            'looks at the feat_colloc_filter column of the specified colloc table')
    group.add_argument('--add_lex_table', action='store_true', dest='addlextable',
                       help='add a lexicon-based feature table. (uses: l, weighted_lexicon, can flag: anscombe).')
    group.add_argument('--add_phrase_table', action='store_true', dest='addphrasetable',
                       help='add constituent and phrase feature tables. (can flag: sqrt, anscombe).')
    group.add_argument('--add_pos_table', action='store_true', dest='addpostable',
                       help='add pos feature tables. (can flag: sqrt, anscombe).')
    group.add_argument('--add_pos_ngram_table', action='store_true', dest='pos_ngram',
                       help='add pos with ngrams feature table. (can flag: sqrt, anscombe).')
    group.add_argument('--add_lda_table', metavar='LDA_MSG_TABLE', dest='addldafeattable',
                       help='add lda feature tables. (can flag: sqrt, anscombe).')
    group.add_argument('--add_tokenized', action='store_true', dest='addtokenized', 
                       help='adds tokenized version of message table.')
    group.add_argument('--add_sent_tokenized', action='store_true', dest='addsenttokenized', 
                       help='adds sentence tokenized version of message table.')
    group.add_argument('--add_parses', action='store_true', dest='addparses', 
                       help='adds parsed versions of message table.')
    group.add_argument('--add_segmented', action="store_true", dest='addsegmented', default=False,
                       help='adds segmented versions of message table.')
    group.add_argument('--segmentation_model',type=str, dest='segmentationModel', default="ctb",
                       help='Chooses which model to use for message segmentation (CTB or PKU; Default CTB)')
    group.add_argument('--add_tweettok', action='store_true', dest='addtweettok', 
                       help='adds tweetNLP tokenized versions of message table.')
    group.add_argument('--add_tweetpos', action='store_true', dest='addtweetpos', 
                       help='adds tweetNLP pos tagged versions of message table.')
    group.add_argument('--add_lda_messages', metavar='LDA_States_File', dest='addldamsgs', 
                       help='add lda topic version of message table.')
    group.add_argument('--add_outcome_feats', action='store_true', dest='addoutcomefeats',
                       help='add a feature table from the specified outcome table.')
    group.add_argument('--add_topiclex_from_topicfile', action='store_true', dest='addtopiclexfromtopicfile',
                       help='creates a lexicon from a topic file, requires --topic_file --topic_lexicon --lex_thresh --topic_lex_method')
    group.add_argument('--print_tokenized_lines', metavar="FILENAME", dest='printtokenizedlines', default = None,
                       help='prints tokenized version of messages to lines.')
    group.add_argument('--print_joined_feature_lines', metavar="FILENAME", dest='printjoinedfeaturelines', default = None,
                       help='prints feature table with line per group joined by spaces (with MWEs joined by underscores) for import into Mallet.')
    group.add_argument('--add_timexdiff', action='store_true', dest='addtimexdiff',
                       help='extract timeex difference features (mean and std) per group.')
    group.add_argument('--add_postimexdiff', action='store_true', dest='addpostimexdiff',
                       help='extract timeex difference features and POS tags per group.')
    group.add_argument('--add_wn_nopos', action='store_true', dest='addwnnopos',
                       help='extract WordNet concept features (not considering POS) per group.')
    group.add_argument('--add_wn_pos', action='store_true', dest='addwnpos',
                       help='extract WordNet concept features (considering POS) per group.')
    group.add_argument('--add_fleschkincaid', '--add_fkscore', action='store_true', dest='addfkscore',
                       help='add flesch-kincaid scores, averaged per group.')
    group.add_argument('--add_pnames', type=str, nargs=2, dest='addpnames',
                       help='add an people names feature table. (two agrs: NAMES_LEX, ENGLISH_LEX, can flag: sqrt)')

    group = parser.add_argument_group('Semantic Extraction Actions', '')
    group.add_argument('--add_ner', action='store_true', dest='addner',
                       help='extract ner features from xml files (corptable should be directory of xml files).')


    group = parser.add_argument_group('Feature Table Aanalyses', '')
    group.add_argument('--ttest_feat_tables', action='store_true', dest='ttestfeats',
                       help='Performs ttest on differences between group norms for 2 tables, within features')

    group = parser.add_argument_group('Refinement Actions', '')
    group.add_argument('--feat_occ_filter', action='store_true', dest='featoccfilter',
                       help='remove infrequent features. (uses variables feat_table and p_occ).')
    group.add_argument('--combine_feat_tables', type=str, dest='combinefeattables', default=None,
                       help='Given multiuple feature table, combines them (provide feature name) ')
    group.add_argument('--add_feat_norms', action='store_true', dest='addfeatnorms',
                       help='calculates and adds the mean normalized (feat_norm) value for each row (uses variable feat_table).')
    group.add_argument('--feat_colloc_filter', action='store_true', dest='featcollocfilter',
                       help='removes featrues that do not pass as collocations. (uses feat_table).')
    group.add_argument('--feat_correl_filter', action='store_true', dest='featcorrelfilter',
                       help='removes features that do not pass correlation sig tests with given outcomes (uses -f --outcome_table --outcomes).')
    group.add_argument('--make_topic_labelmap_lex', action='store_true', dest='maketopiclabelmap', default=False,
                       help='Makes labelmap lexicon from topics. Requires --topic_lexicon, --num_topic_words. Optional: --weighted_lexicon')
    group.add_argument('--feat_group_by_outcomes', action='store_true', dest='featgroupoutcomes', default=False,
                       help='Creates a feature table grouped by a given outcome (requires outcome field, can use controls)')
    group.add_argument('--aggregate_feats_by_new_group', action='store_true', dest='aggregategroup', default=False,
                       help='Aggregate feature table by group field (i.e. message_id features by user_ids).')

    group = parser.add_argument_group('Outcome Actions', '')
    group.add_argument('--print_csv', metavar="FILENAME", dest='printcsv', default = None,
                       help='prints group normalized values use for correlation.')
    group.add_argument('--print_freq_csv', metavar="FILENAME", dest='printfreqcsv', default = None,
                       help='prints frequencies instead of group normalized values')
    group.add_argument('--print_numgroups', action='store_true', dest='printnumgroups', default = False,
                       help='prints number of groups per outcome field')
    group.add_argument('--notify', type=str, dest='notify', default='',
                       help='sends a completion email to the designated email address')
    group.add_argument('--notify_luke', type=str, dest='notifyluke', default='',
                       help='sends a completion email to luke with the specified message. If no message is specified, no email is sent')
    group.add_argument('--notify_andy', type=str, dest='notifyandy', default='',
                       help='sends a completion email to andy with the specified message. If no message is specified, no email is sent')
    group.add_argument('--notify_johannes', type=str, dest='notifyjohannes', default='',
                       help='sends a completion email to johannes with the specified message. If no message is specified, no email is sent')

    group = parser.add_argument_group('Correlation Actions', 'Finds one relationship at a time (but can still adjust for others)')
    group.add_argument('--correlate', action='store_true', dest='correlate',
                       help='correlate with outcome (uses variable feat_table and all outcome variables).')
    group.add_argument('--rmatrix', action='store_true', dest='rmatrix',
                       help='output a correlation matrix to a file in the output dir.')
    group.add_argument('--combo_rmatrix', action='store_true', dest='combormatrix',
                       help='output a correlation matrix with all combinations of controls.')
    group.add_argument('--topic_dupe_filter', action='store_true', dest='topicdupefilter',
                       help='remove topics not passing a duplicate filter from the correlation matrix')
    group.add_argument('--tagcloud', action='store_true', dest='tagcloud',
                       help='produce data for making wordle tag clouds (same variables as correlate).')
    group.add_argument('--topic_tagcloud', action='store_true', dest='topictc', 
                       help='produce data for making topic wordles (must be used with a topic-based feature table and --topic_lexicon).')
    group.add_argument('--make_wordclouds', action='store_true', dest='makewordclouds',
                       help="make wordclouds from the output tagcloud file.")
    group.add_argument('--make_topic_wordclouds', action='store_true', dest='maketopicwordclouds',
                       help="make topic wordclouds, needs an output topic tagcloud file.")
    group.add_argument('--use_featuretable_feats', action='store_true', dest='useFeatTableFeats',
                       help='use 1gram table to be used as a whitelist when plotting')
    group.add_argument('--outcome_with_outcome', action='store_true', dest='outcomeWithOutcome',
                       help="correlates all outcomes in --outcomes with each other in addition to the features")
    group.add_argument('--output_interaction_terms', action='store_true', dest='outputInteractionTerms',
                       help='with this flag, outputs the coefficients from the interaction terms as r values '+
                       'the outcome coefficients. Use with --outcome_interaction FIELD1 [FIELD2 ...]')
    group.add_argument('--interaction_ddla', dest='interactionDdla',
                       help="column name from the outcome table that is going to be used in DDLA:"+
                       "First, finding terms with significant interaction, then taking correlations for groups with outcome = 1 and = 0 separately")
    group.add_argument('--interaction_ddla_pvalue', dest='ddlaSignificance', type=float,default=0.001,
                       help="Set level of significance to filter ddla features by")
    group.add_argument('--DDLA', dest='ddlaFiles', nargs=2, help="Compares two csv's that have come out of DLA. Requires --freq and --nvalue to have been used")
    group.add_argument('--DDLATagcloud', dest='ddlaTagcloud', action='store_true',
                       help="Makes a tagcloud file from the DDLA output. Uses deltaR as size, r_INDEX as color. ")


    group = parser.add_argument_group('Multiple Regression Actions', 'Find multiple relationships at once')
    group.add_argument('--multir', action='store_true', dest='multir',
                       help='multivariate regression with outcome (uses variable feat_table and all outcome variables, optionally csv).')

    group = parser.add_argument_group('Prediction Actions', '')
    group.add_argument('--train_regression', '--train_reg', action='store_true', dest='trainregression', default=False,
                       help='train a regression model to predict outcomes based on feature table')
    group.add_argument('--test_regression', action='store_true', dest='testregression', default=False,
                       help='train/test a regression model to predict outcomes based on feature table')
    group.add_argument('--combo_test_regression', '--combo_test_reg', action='store_true', dest='combotestregression', default=False,
                       help='train/test a regression model with and without all combinations of controls')
    group.add_argument('--predict_regression', '--predict_reg', action='store_true', dest='predictregression', default=False,
                       help='predict outcomes based on loaded or trained regression model')
    group.add_argument('--control_adjust_outcomes_regression', '--control_adjust_reg', action='store_true', default=False,  dest='controladjustreg', 
                       help='predict outcomes from controls and produce adjusted outcomes')
    group.add_argument('--test_combined_regression', type=str, metavar="featuretable", nargs='+', dest='testcombregression', default=[],
                       help='train and test combined model (must specify at least one addition feature table here)')
    group.add_argument('--predict_regression_to_feats', type=str, dest='predictrtofeats', default=None,
                       help='predict outcomes into a feature file (provide a name)')
    group.add_argument('--predict_regression_to_outcome_table', type=str, dest='predictRtoOutcomeTable', default=None,
                       help='predict outcomes into an outcome table (provide a name)')
    group.add_argument('--predict_cv_to_feats', '--predict_combo_to_feats', '--predict_regression_all_to_feats', type=str, dest='predictalltofeats', default=None,
                       help='predict outcomes into a feature file (provide a name)')

    group.add_argument('--train_classifiers', '--train_class', action='store_true', dest='trainclassifiers', default=False,
                       help='train classification models for each outcome field based on feature table')
    group.add_argument('--test_classifiers', action='store_true', dest='testclassifiers', default=False,
                       help='trains and tests classification for each outcome')
    group.add_argument('--combo_test_classifiers', action='store_true', dest='combotestclassifiers', default=False,
                       help='train/test a regression model with and without all combinations of controls')
    group.add_argument('--predict_classifiers', '--predict_class', action='store_true', dest='predictclassifiers', default=False,
                       help='predict outcomes bases on loaded training')
    group.add_argument('--roc', action='store_true', dest='roc',
                       help="Computes ROC curves and outputs to PDF")
    group.add_argument('--predict_classifiers_to_feats', type=str, dest='predictctofeats', default=None,
                       help='predict outcomes into a feature file (provide a name)')
    group.add_argument('--predict_classifiers_to_outcome_table', type=str, dest='predictCtoOutcomeTable', default=None,
                       help='predict outcomes into an outcome table (provide a name)')
    group.add_argument('--regression_to_lexicon', dest='regrToLex', type=str, default=None,
                       help='Uses the regression coefficients to create a weighted lexicon.')
    group.add_argument('--classification_to_lexicon', dest='classToLex', type=str, default=None,
                       help='Uses the classification coefficients to create a weighted lexicon.')
    group.add_argument('--reducer_to_lexicon', type=str, dest='reducertolexicon', default=None,
                       help='writes the reduction model to a specified lexicon')
    group.add_argument('--fit_reducer', action='store_true', dest='fitreducer', default=False,
                       help='reduces a feature space to clusters')
    group.add_argument('--cca', type=int, dest='cca', default=0,
                       help='Performs sparse CCA on a set of features and a set of outcomes.'+
                       "Argument is number of components to output (Uses R's PMA package)")
    group.add_argument('--cca_penalty_feats', '--cca_penaltyx', type=float, dest='penaltyFeats', default = None,
                       help="Penalty value on the feature matrix (X) [penaltyx argument of PMA.CCA]"+
                       "must be between 0 and 1, larger means less penalization (i.e. less sparse) ")
    group.add_argument('--cca_penalty_outcomes', '--cca_penaltyz', type=float, dest='penaltyOutcomes', default = None,
                       help="Penalty value on the outcomes matrix (Z) [penaltyz argument of PMA.CCA]"+
                       "must be between 0 and 1, larger means less penalization (i.e. less sparse) ")
    group.add_argument('--cca_outcomes_vs_controls', dest='ccaOutcomesVsControls',action='store_true',
                       help="performs CCA on outcomes vs controls (no language)")
    group.add_argument('--cca_permute', dest='ccaPermute', type=int,default=0,
                       help='Wrapper for the PMA package CCA.permute function that determines the'+
                       ' ideal L1 Penalties for X and Z matrices'+
                       'argument: number of permutations')
    group.add_argument('--cca_predict_components', dest='predictCcaCompsFromModel',action="store_true",
                       help='Using --picklefile, predict outcomes from the V matrix (aka Z_comp)')
    group.add_argument('--to_sql_table', dest='newSQLtable',type=str,
                       help='Using --cca_predict_components, predict components to sql table,'+
                       'the name of which you should provide here')

    group.add_argument('--train_c2r', action='store_true', dest='trainclasstoreg', default=False,
                       help='train a model that goes from classification to prediction')
    group.add_argument('--test_c2r', action='store_true', dest='testclasstoreg', default=False,
                       help='trains and tests classification for each outcome')
    group.add_argument('--predict_c2r', action='store_true', dest='predictclasstoreg', default=False,
                       help='predict w/ classification to regression model')


    group.add_argument('--save_models', action='store_true', dest='savemodels', default=False,
                       help='saves predictive models (uses --picklefile)')
    group.add_argument('--load_models', action='store_true', dest='loadmodels', default=False,
                       help='loads predictive models (uses --picklefile)')


    group = parser.add_argument_group('Plot Actions', '')
    group.add_argument('--barplot', action='store_true', dest='barplot',
                       help='produce correlation barplots. Requires fg, oa. Uses groupfreqthresh, outputdir')
    group.add_argument('--scatterplot', action='store_true', dest='scatterplot',
                       help='Requires --outcome_table --outcome_fields, optional: -f --feature_names')
    group.add_argument('--feat_flexibin', action='store_true', dest='featflexibin', default=False,
                       help='Plots a binned feature table, uses --num_bins, --group_id_range, --feat_table, --flexiplot_file')    # group.add_argument('--hist2d', action='store_true', dest='hist2d',
    group.add_argument('--skip_bin_step', action='store_true', dest='skipbinstep', default=False,
                       help='Skips the binning step for feat_flexibin. For when we want fast plotting and the flexitable has been created.')    
    group.add_argument('--preserve_bin_table', action='store_true', dest='preservebintable', default=False,
                       help='Preserves the flexibin table for faster plotting.')    
    # group.add_argument('--hist    
    # group.add_argument('--hist2d', action='store_true', dest='hist2d',
    #                    help='Requires -f --feature_names --outcome_table --outcome_value')
    group.add_argument('--descplot', action='store_true', dest='descplot',
                       help='produce histograms and boxplots for specified outcomes. Requires oa. Uses outputdir')
    group.add_argument('--loessplot', type=str, metavar='FEAT(S)', dest='loessplot', nargs='+', default='',
                       help='Output loess plots of the given features.')
    group.add_argument('--v2', action='store_true', dest='v2',
                       help='Run commands from other place')

    if fn_args:
        args = parser.parse_args(fn_args.split())
    else:
        args = parser.parse_args()

    print args
    print args.toinitfile, args.frominitfile
    #exit()
    # #options = vars(args) #pass to function to avoid passing lots of arguments
    ##NON-Specified Defaults:

    if args.v2:
        from PERMA.code.fwv2 import wwbp
        if fn_args:
            wwbp.main(fn_args)
        else:
            wwbp.main(" ".join(sys.argv[1:]))

    ##Argument adjustments: 
    if not args.valuefunc: args.valuefunc = lambda d: d
    if not args.lexvaluefunc: args.lexvaluefunc = lambda d: d
    if args.bonferroni: args.simes = False
    if args.p_correction_method: args.bonferroni = False

    if args.feattable and len(args.feattable) == 1:
        args.feattable = args.feattable[0]

    if not args.feattable and args.aggregategroup:
        args.feattable = aggregategroup[0]

    if args.weightedeval:
        args.outcomefields.append(args.weightedeval)

    # DEF_LEXICON_DB = args.lexicondb
    FeatureWorker.lexicon_db = args.lexicondb

    ##Process Arguments
    def FE():
        return FeatureExtractor(args.corpdb, args.corptable, args.correl_field, args.mysql_host, args.message_field, args.messageid_field, args.lexicondb, wordTable = args.wordTable)

    def SE():
        return SemanticsExtractor(args.corpdb, args.corptable, args.correl_field, args.mysql_host, args.message_field, args.messageid_field, args.lexicondb, args.corpdir, wordTable = args.wordTable)

    def OG():
        return OutcomeGetter(args.corpdb, args.corptable, args.correl_field, args.mysql_host, args.message_field, args.messageid_field, args.lexicondb, args.outcometable, args.outcomefields, args.outcomecontrols, args.outcomeinteraction, args.featlabelmaptable, args.featlabelmaplex, wordTable = args.wordTable)

    def OA():
        return OutcomeAnalyzer(args.corpdb, args.corptable, args.correl_field, args.mysql_host, args.message_field, args.messageid_field, args.lexicondb, args.outcometable, args.outcomefields, args.outcomecontrols, args.outcomeinteraction, args.featlabelmaptable, args.featlabelmaplex, wordTable = args.wordTable, output_name = args.outputname)

    def FR():
        return FeatureRefiner(args.corpdb, args.corptable, args.correl_field, args.mysql_host, args.message_field, args.messageid_field, args.lexicondb, args.feattable, args.featnames, wordTable = args.wordTable)

    def FG(featTable = None):
        if not featTable:
            featTable = args.feattable
        return FeatureGetter(args.corpdb, args.corptable, args.correl_field, args.mysql_host, args.message_field, args.messageid_field, args.lexicondb, featTable, args.featnames, wordTable = args.wordTable)

    def FGs(featTable = None):
        if not featTable:
            featTable = args.feattable
            if not featTable:
                print "Need to specify feature table(s)"
                sys.exit(1)
        if isinstance(featTable, basestring):
            featTable = [featTable]
        return [FeatureGetter(args.corpdb,
                              args.corptable,
                              args.correl_field,
                              args.mysql_host,
                              args.message_field,
                              args.messageid_field,
                              args.lexicondb, featTable,
                              args.featnames,
                              wordTable = args.wordTable)
                for featTable in featTable]


    fe = None
    se = None
    fr = None
    og = None
    oa = None
    fg = None
    fgs = None #feature getters

    #Feature Extraction:
    if args.addngrams:
        if not fe: fe = FE()

        if args.use_collocs:
            pmi_filter_thresh = args.colloc_pmi_thresh if args.colloc_pmi_thresh else DEF_PMI
            collocs_list = fe._getCollocsFromTable(args.colloc_table, pmi_filter_thresh, args.colloc_column, DEF_COLUMN_PMI_FILTER)
            if args.feature_type_name:
                feature_type_name = args.feature_type_name
            else:
#                feature_type_name = "clc" + str(pmi_filter_thresh).replace('.', '_')
                feature_type_name = "colloc"
            args.feattable = fe.addCollocFeatTable(collocs_list, valueFunc = args.valuefunc, includeSubCollocs=args.include_sub_collocs, featureTypeName = feature_type_name)

        elif args.gzipcsv:
            args.feattable = fe.addNGramTableGzipCsv(args.n, args.gzipcsv, 3, 0, 19, valueFunc = args.valuefunc)

        else:
            ftables = list()
            for n in args.n:
                ftables.append(fe.addNGramTable(n, valueFunc = args.valuefunc, metaFeatures = args.metafeats))
            if len(ftables) > 1:
                args.feattable = ftables;
            else:
                args.feattable = ftables[0]
    if args.addngramsfromtok:
        if not fe: fe = FE()
        ftables = list()
        for n in args.n:
            ftables.append(fe.addNGramTableFromTok(n, valueFunc = args.valuefunc, metaFeatures = args.metafeats))
            if len(ftables) > 1:
                args.feattable = ftables;
            else:
                args.feattable = ftables[0]
    if args.addlextable:
        if not fe: fe = FE()
        args.feattable = fe.addLexiconFeat(args.lextable, valueFunc = args.valuefunc, isWeighted=args.weightedlexicon, featValueFunc=args.lexvaluefunc)

    if args.addphrasetable:
        if not fe: fe = FE()
        args.feattable = fe.addPhraseTable(valueFunc = args.valuefunc)

    if args.addpostable or args.pos_ngram:
        if not fe: fe = FE()
        args.feattable = fe.addPosTable(valueFunc = args.valuefunc, keep_words = args.pos_ngram)

    if args.addsegmented:
        if not fe: fe = FE()
        fe.addSegmentedMessages(args.segmentationModel)

    if args.addldafeattable:
        if not fe: fe = FE()
        args.feattable = fe.addLDAFeatTable(args.addldafeattable, valueFunc = args.valuefunc)

    if args.addpnames:
        if not fe: fe = FE()
        namesLex = lexInterface.Lexicon(mysql_host = args.mysql_host)
        namesLex.loadLexicon(args.addpnames[0])
        langLex = lexInterface.Lexicon(mysql_host = args.mysql_host)
        langLex.loadLexicon(args.addpnames[1])
        args.feattable = fe.addPNamesTable(namesLex.getLexicon(), langLex.getLexicon(),  valueFunc = args.valuefunc)

    if args.addwnnopos:
        if not fe: fe = FE()
        args.feattable = fe.addWNNoPosFeat(valueFunc = args.valuefunc, featValueFunc=args.lexvaluefunc)
        
    if args.addwnpos:
        if not fe: fe = FE()
        args.feattable = fe.addWNPosFeat(pos_table = args.feattable, valueFunc = args.valuefunc, featValueFunc=args.lexvaluefunc)

    if args.addfkscore:
        if not fe: fe = FE()
        args.feattable = fe.addFleschKincaidTable(valueFunc = args.valuefunc)



    if args.addtokenized:
        if not fe: fe = FE()
        fe.addTokenizedMessages()

    if args.addsenttokenized:
        if not fe: fe = FE()
        fe.addSentTokenizedMessages()

    if args.printtokenizedlines:
        if not fe: fe = FE()
        if args.feat_whitelist:
            fe.printTokenizedLines(args.printtokenizedlines, whiteListFeatTable = args.feat_whitelist)
            #TODO: change whitelistfeat table to get a list instead. 
        else:
            fe.printTokenizedLines(args.printtokenizedlines)

    if args.printjoinedfeaturelines:
        if not fg: fg = FG()
        fg.printJoinedFeatureLines(args.printjoinedfeaturelines)
      
    if args.addparses:
        if not fe: fe = FE()
        fe.addParsedMessages()

    if args.addtweettok:
        if not fe: fe = FE()
        fe.addTweetTokenizedMessages()

    if args.addtweetpos:
        if not fe: fe = FE()
        fe.addTweetPOSMessages()

    if args.addldamsgs:
        if not fe: fe = FE()
        fe.addLDAMessages(args.addldamsgs)

    if args.addtopiclexfromtopicfile:
        if not fe: fe = FE()
        if not args.topicfile or not args.topiclexicon or not args.topiclexmethod:
            raise Exception('cannot add topic lexicon from topic file without specifying --topic_file, --topic_lexicon, --topic_lex_method')
#--topic_file --topic_lexicon --lex_thresh --topic_method')
        if args.lexthresh == None: args.lexthresh = float('-inf')
        fe.addTopicLexFromTopicFile(args.topicfile, args.topiclexicon, args.topiclexmethod, args.lexthresh)

    if args.addoutcomefeats:
        if not fe: fe = FE()
        if not og: og = OG()
        fe.addOutcomeFeatTable(og, valueFunc = args.valuefunc)

    if args.addtimexdiff:
        if not fe: fe = FE()
        args.feattable = fe.addTimexDiffFeatTable(args.date_field)

    if args.addpostimexdiff:
        if not fe: fe = FE()
        args.feattable = fe.addPOSAndTimexDiffFeatTable(args.date_field, valueFunc = args.valuefunc)

    #Semantic Feature Extraction:
    if args.addner:
        if not se: se = SE()
        args.feattable = se.addNERTable(valueFunc = args.valuefunc)


    

    #Feature Refinement:

    #(first do refinements that group tables so these tables can then be refined further by filter refinements)
    if args.combinefeattables:
        if not fr: fr=FR()
        args.feattable = fr.createCombinedFeatureTable(args.combinefeattables, args.feattable) 
        #TODO: use internal fr variable for feature tables rather than argument
        fr = None #so that the feature table must be re-taken

    if args.featgroupoutcomes:
        if not fr: fr=FR()
        if not og: og=OG()
        args.feattable = fr.createFeatTableByDistinctOutcomes(og, nameSuffix=args.outputname)

    if args.aggregategroup:
        if not fr: fr=FR()
        args.feattable = fr.createAggregateFeatTableByGroup(valueFunc = args.valuefunc)

    if args.featoccfilter:
        if args.use_collocs and not args.wordTable:
            args.wordTable = args.feattable
        if not fr: fr=FR()
        args.feattable = fr.createTableWithRemovedFeats(args.pocc,args.minfeatsum,args.groupfreqthresh)

    if args.featcollocfilter:
        if not fr: fr=FR()
        if args.use_collocs:
            raise NotImplementedError
        # args.feattable = fr.createCollocRefinedFeatTable(featNormTable=False) #faster ## FEAT NORM TABLE LUKASZ LUKE
        args.feattable = fr.createCollocRefinedFeatTable(args.pmi)

    if args.maketopiclabelmap:
        if not fr: fr=FR()
        args.featlabelmaplex = fr.makeTopicLabelMap(args.topiclexicon, args.numtopicwords, args.weightedlexicon)

    if args.featflexibin:
        if not fr: fr=FR()
        if not args.numbins or not args.groupidrange or not args.feattable:
            raise Exception('cannot bin feature table without specifying --num_bins, --group_id_range, --feat_table') #optionally specify --topic_lexicon
        args.feattable = fr.createTableWithBinnedFeats(args.numbins, args.groupidrange, args.groupfreqthresh, args.valuefunc, args.gender, args.genderattack, skip_binning=args.skipbinstep)#, args.reportingpercent)
        # sys.exit(0)
        # need feature csv for batch plotting
        temp_feature_file = OutcomeGetter.buildBatchPlotFile(args.corpdb, args.feattable, args.topiclist) if not args.flexiplotfile else args.flexiplotfile
        feat_to_label = None
        if args.topiclexicon:
            if not oa: oa=OA()
            feat_to_label = oa.buildTopicLabelDict(args.topiclexicon, 10) #TODO-finish -- uses topictagcloudwords method
            #pprint(feat_to_label)
        OutcomeGetter.plotFlexibinnedTable(args.corpdb, args.feattable, temp_feature_file, feat_to_label, args.preservebintable)
       

    #if args.addmean: #works, but excessive option
    #    if not fr: fr=FR()
    #    fr.addFeatTableMeans()

    if args.addfeatnorms:
        if not fr: fr=FR()
        fr.addFeatNorms()

    #create whitelist / blacklist
    if args.categories:
        if isinstance(args.categories, basestring):
            args.categories = [args.categories]
    if args.feat_blacklist:
        if isinstance(args.feat_blacklist, basestring):
            args.feat_blacklist = [args.feat_blacklist]
    if args.feat_whitelist:
        if isinstance(args.feat_whitelist, basestring):
            args.feat_whitelist = [args.feat_whitelist] 
    (whitelist, blacklist) = (None, None)
    # Wildcards are not handled!!!
    if args.blacklist:
        blacklist = FeatureWorker.makeBlackWhiteList(args.feat_blacklist, args.lextable, args.categories, args.lexicondb)
    if args.whitelist:
        whitelist = FeatureWorker.makeBlackWhiteList(args.feat_whitelist, args.lextable, args.categories, args.lexicondb)

    def makeOutputFilename(args, fg=None, og=None, prefix=None, suffix=None):
        if args.outputname:
            return args.outputname+suffix

        outputFile = args.outputdir + '/'
        filePieces = []
        if prefix:            
            filePieces.append(prefix)
        if fg:
            filePieces.append(fg.featureTable)
        if og:
            filePieces.append(og.outcome_table)
            if og.outcome_value_fields and len(og.outcome_value_fields) <= 10:
                filePieces.append('_'.join(og.outcome_value_fields))
        if og.outcome_controls: filePieces.append('_'.join(og.outcome_controls))
        if args.groupfreqthresh: filePieces.append("freq" + str(args.groupfreqthresh))
        if args.spearman: filePieces.append('spear')
        if args.blacklist: filePieces.append('blist')
        if args.lextable: filePieces.append('-'.join([args.lextable] + args.categories))
        if args.feat_blacklist: filePieces.append('-'.join(args.feat_blacklist))
        if args.feat_whitelist: filePieces.append('wlist')
        if args.feat_whitelist: filePieces.append('-'.join(args.feat_whitelist))
        if suffix: filePieces.append(suffix)
        outputFile += '.'.join(filePieces)
        outputFile = outputFile.replace("$", ".")
        print "Created output filename: %s" % outputFile
        return outputFile

    #Outcome Only options:
    if args.printcsv:
        pprint(args)
        if not oa: oa = OA()
        if not fg: fg = FG()
#        if args.isbinned:
#            og.printBinnedGroupsAndOutcomesToCSV(fg, args.printcsv, args.groupfreqthresh)
#        else:
        oa.printGroupsAndOutcomesToCSV(fg, args.printcsv, groupThresh=args.groupfreqthresh)

    if args.printfreqcsv:
        pprint(args)
        if not oa: oa = OA()
        if not fg: fg = FG()
        oa.printGroupsAndOutcomesToCSV(fg, args.printcsv, args.groupfreqthresh, freqs = True)

    if args.printnumgroups:
        pprint(args)
        if not oa: oa = OA()
        if not fg: fg = FG()
        pprint(('number of groups per outcome:', oa.numGroupsPerOutcome(fg, args.printcsv, args.groupfreqthresh)))

    if args.loessplot:
        if not oa: oa = OA()
        if not fg: fg = FG()
        # whitelist = whitelist
        oa.loessPlotFeaturesByOutcome(fg, args.groupfreqthresh, args.spearman, args.bonferroni, args.p_correction_method, blacklist, whitelist.union(args.loessplot), args.showfeatfreqs, outputdir=args.outputdir, outputname=args.outputname, topicLexicon=args.topiclexicon)        

    #Correlation Analysis Options:
    correls = None
    if args.compTagcloud and args.compTCsample1 and args.compTCsample2:
        if not fg: fg = FG()
        if not oa: oa = OA()
        correls = oa.IDPcomparison(fg, args.compTCsample1, args.compTCsample2, groupThresh=args.groupfreqthresh, blacklist=blacklist, whitelist=whitelist)

    if not args.compTagcloud and not args.cca and (args.correlate or args.rmatrix or args.tagcloud or args.topictc or args.barplot or args.featcorrelfilter or args.makewordclouds or args.maketopicwordclouds):
        if not oa: oa = OA()
        if not fg: fg = FG()
        if args.interactionDdla:
            # if len(args.outcomefieldsprint "There were no features with significant interactions") > 1: raise NotImplementedError("Multiple outcomes with DDLA not yet implemented")
            # Step 1 Interaction
            args.outcomeinteraction = [args.interactionDdla]
            oa = OA()
            print "##### STEP 1: Finding features with significant interaction term"
            correls = oa.correlateWithFeatures(fg, args.groupfreqthresh, args.spearman, args.bonferroni,
                                               args.p_correction_method, args.outcomeinteraction, blacklist,
                                               whitelist, args.showfeatfreqs, args.outcomeWithOutcome,
                                               logisticReg=args.logisticReg, outputInteraction=True)
            inter_keys = [i for i in correls.keys() if " * " in i]
            # correls = {outcome1: {feat: (R,p,N,freq)}}
            
            # whitelist should be different for multiple outcomes
            ddla_whitelists = {inter_key: [k for k, i in correls[inter_key].iteritems() if i[1] < args.ddlaSignificance] for inter_key in inter_keys}
            print "Maarten", ddla_whitelists

            correls = {"INTER["+k+"]": v for k, v in correls.iteritems()}
            
            for out_name, ddla_whitelist in ddla_whitelists.iteritems():
                if not ddla_whitelist:
                    continue
                out = out_name.split(" from ")[-1]
                print "Maarten", out_name, out
                
                whitelist = FeatureWorker.makeBlackWhiteList(ddla_whitelist, '', [], args.lexicondb)                
                
                # print str([i for j in correls.values() for i in j.iteritems() if i[1][0]*i[1][0] > 1])[:300]
                # exit()
                # Step 2: do correlations on both ends of the interaction variable
                
                print "##### STEP 2: getting correlations within groups"
                print "args.outcomecontrols", args
                args.outcomeinteraction = []
                args.outcomefields = [out]
                og = OG()
                where = args.interactionDdla+"=1"
                correls_1 = oa.correlateWithFeatures(fg, args.groupfreqthresh, args.spearman, args.bonferroni,
                                                     args.p_correction_method, args.outcomeinteraction, blacklist,
                                                     whitelist, args.showfeatfreqs, args.outcomeWithOutcome,
                                                     logisticReg=args.logisticReg, groupWhere = where)
                
                correls.update({"["+k+"]_1": v for k, v in correls_1.iteritems()})
                og = OG()
                where = args.interactionDdla+"=0"
                correls_0 = oa.correlateWithFeatures(fg, args.groupfreqthresh, args.spearman, args.bonferroni,
                                                     args.p_correction_method, args.outcomeinteraction, blacklist,
                                                     whitelist, args.showfeatfreqs, args.outcomeWithOutcome,
                                                     logisticReg=args.logisticReg, groupWhere = where)
                correls.update({"["+k+"]_0": v for k, v in correls_0.iteritems()})

        elif args.IDP:        
            correls = oa.IDP_correlate(fg, groupThresh=args.groupfreqthresh, outcomeWithOutcome=args.outcomeWithOutcome, includeFreqs=args.showfeatfreqs,blacklist=blacklist, whitelist=whitelist ) 
        elif args.zScoreGroup:
            correls = oa.zScoreGroup(fg, groupThresh=args.groupfreqthresh, outcomeWithOutcome=args.outcomeWithOutcome, includeFreqs=args.showfeatfreqs, blacklist=blacklist, whitelist=whitelist)
        elif args.auc:        
            correls = oa.aucWithFeatures(fg, groupThresh=args.groupfreqthresh, bonferroni = args.bonferroni, outcomeWithOutcome=args.outcomeWithOutcome, includeFreqs=args.showfeatfreqs,blacklist=blacklist, whitelist=whitelist, bootstrapP = args.bootstrapp) 
        else:
            correls = oa.correlateWithFeatures(fg, args.groupfreqthresh, args.spearman, args.bonferroni, args.p_correction_method, args.outcomeinteraction, blacklist, whitelist, args.showfeatfreqs, args.outcomeWithOutcome, logisticReg=args.logisticReg, outputInteraction=args.outputInteractionTerms)
        if args.topicdupefilter:#remove duplicate topics (keeps those correlated more strongly)
            correls = oa.topicDupeFilterCorrels(correls, args.topiclexicon)

    if args.ccaPermute:
        if not og: og = OG()
        if not fg: fg = FG()
        cca = CCA(fg, og, args.cca)
        if args.ccaOutcomesVsControls:
            cca.ccaPermuteOutcomesVsControls(args.groupfreqthresh, nPerms = args.ccaPermute)
        else:
            cca.ccaPermute(args.groupfreqthresh, nPerms = args.ccaPermute)
        
    if args.predictCcaCompsFromModel:
        if not og: og = OG()
        if not fg: fg = FG()
        cca = CCA(fg, og)
        if args.loadmodels:
            cca.loadModel(args.picklefile)
        components = cca.predictCompsToSQL(tablename=args.newSQLtable,
                                           groupFreqThresh = args.groupfreqthresh,
                                           csv = args.csv,
                                           outputname = args.outputname if args.outputname
                                           else args.outputdir + '/rMatrix.' + fg.featureTable + '.' + og.outcome_table  + '.' + '_'.join(og.outcome_value_fields))
    if args.cca:
        if not oa: oa = OA()
        if not fg: fg = FG()
        cca = CCA(fg, oa, args.cca)
        if args.ccaOutcomesVsControls:
            (featComp, outcomeComp, dVectorDict) = cca.ccaOutcomesVsControls(args.groupfreqthresh,
                                                                penaltyX = args.penaltyFeats,
                                                                penaltyZ = args.penaltyOutcomes)
        else:
            (featComp, outcomeComp, dVectorDict) = cca.cca(args.groupfreqthresh,
                                                           penaltyX = args.penaltyFeats,
                                                           penaltyZ = args.penaltyOutcomes)

        paramString = str(args)+"\n\n"
        paramString += "Components D vector\n"+'\n'.join(["%s, %.3f" % (comp,d) for comp, d in sorted(dVectorDict.items())])
        correls = featComp
        # Print csvs, topic_tagcloud
        if args.rmatrix: 
            if args.outputname:
                outputFile = args.outputname
            else: 
                outputFile = args.outputdir + '/rMatrix.' + fg.featureTable + '.' + oa.outcome_table  + '.' + '_'.join(oa.outcome_value_fields)
                if oa.outcome_controls: outputFile += '.'+ '_'.join(oa.outcome_controls)
                if args.spearman: outputFile += '.spearman'
            oa.correlMatrix(featComp, outputFile+".feat", outputFormat='html', sort=args.sort, paramString=paramString.replace("\n","<br>"), nValue=args.nvalue, freq=args.freq)
            oa.correlMatrix(outcomeComp, outputFile+".outcome", outputFormat='html', sort=args.sort, paramString=paramString.replace("\n","<br>"), nValue=args.nvalue, freq=args.freq)
            if args.csv:
                oa.correlMatrix(featComp, outputFile+".feat", outputFormat='csv', sort=args.sort, paramString=paramString, nValue=args.nvalue, freq=args.freq)
                oa.correlMatrix(outcomeComp, outputFile+".outcome", outputFormat='csv', sort=args.sort, paramString=paramString, nValue=args.nvalue, freq=args.freq)
            if args.pickle:
                oa.correlMatrix(featComp, outputFile+".feat", outputFormat='pickle', sort=args.sort, paramString=paramString, nValue=args.nvalue, freq=args.freq)
                oa.correlMatrix(outcomeComp, outputFile+".outcome", outputFormat='pickle', sort=args.sort, paramString=paramString, nValue=args.nvalue, freq=args.freq)
            outputFile += ".feat"
        if args.savemodels:
            cca.saveModel(args.picklefile)

    if args.correlate:
        pprint(args)
        for outcomeField, featRs in correls.iteritems():
            print "\n%s:" % outcomeField
            cnt = 0
            for featR in featRs.iteritems():
                if featR[1][1] < args.maxP: cnt +=1 
            pprint(sorted(featRs.items(), key= lambda f: f[1] if not isnan(f[1][0]) else 0))
            print "\n%d features significant at p < %s" % (cnt, args.maxP)

    if args.rmatrix and not args.cca: 
        if args.outputname:
            outputFile = args.outputname
        else: 
            outputFile = args.outputdir + '/rMatrix.' + fg.featureTable + '.' + oa.outcome_table  + '.' + '_'.join(oa.outcome_value_fields)
            if oa.outcome_controls: outputFile += '.'+ '_'.join(oa.outcome_controls)
            if args.spearman: outputFile += '.spearman'
        oa.correlMatrix(correls, outputFile, outputFormat='html', sort=args.sort, paramString=str(args), nValue=args.nvalue, freq=args.freq)

    if args.csv and not args.cca and correls:
        if args.outputname:
            outputFile = args.outputname
        else: 
            outputFile = args.outputdir + '/rMatrix.' + fg.featureTable + '.' + oa.outcome_table  + '.' + '_'.join(oa.outcome_value_fields)
            if oa.outcome_controls: outputFile += '.'+ '_'.join(oa.outcome_controls)
            if args.spearman: outputFile += '.spearman'
        oa.correlMatrix(correls, outputFile, outputFormat='csv', sort=args.sort, paramString=str(args), nValue=args.nvalue, freq=args.freq)

    if args.tagcloud:
        outputFile = makeOutputFilename(args, fg, oa, suffix="_tagcloud")
        oa.printTagCloudData(correls, args.maxP, outputFile, str(args), maxWords = args.maxtcwords, duplicateFilter = args.tcfilter, colorScheme=args.tagcloudcolorscheme)
    if args.makewordclouds:
        if not args.tagcloud:
            print >>sys.stderr, "ERROR, can't use --make_wordclouds without --tagcloud"
            sys.exit()
        wordcloud.tagcloudToWordcloud(outputFile, withTitle=True, fontFamily="Meloche Rg", fontStyle="bold", toFolders=True)

    if args.topictc:
        outputFile = makeOutputFilename(args, fg, oa, suffix='_topic_tagcloud')
        # use plottingWhitelistPickle to link to a pickle file containing the words driving the categories
        oa.printTopicTagCloudData(correls, args.topiclexicon, args.maxP, str(args), duplicateFilter = args.tcfilter, colorScheme=args.tagcloudcolorscheme, outputFile = outputFile, useFeatTableFeats=args.useFeatTableFeats)
        # don't want to base on this: maxWords = args.maxtcwords)
    if args.maketopicwordclouds:
        if not args.topictc:
            print >>sys.stderr, "ERROR, can't use --make_topic_wordclouds without --topic_tagcloud"
            sys.exit()
        wordcloud.tagcloudToWordcloud(outputFile, withTitle=True, fontFamily="Meloche Rg", fontStyle="bold", toFolders=True)

    comboCorrels = None
    if args.combormatrix:
        if not oa: oa = OA()
        if not fg: fg = FG()
        comboCorrels = oa.correlateControlCombosWithFeatures(fg, args.groupfreqthresh, args.spearman, args.bonferroni, args.p_correction_method, blacklist, whitelist, args.showfeatfreqs, args.outcomeWithOutcome)
        if args.csv:
            outputStream = sys.stdout
            if args.outputname:
                outputStream = open(args.outputname+'.combocntrl.csv', 'w')
            oa.outputComboCorrelMatrixCSV(comboCorrels, outputStream, paramString = str(args))
        else:
            pprint(comboCorrels)

    if args.featcorrelfilter:
         if not fr: fr=FR()
         args.feattable = fr.createCorrelRefinedFeatTable(correls)
         
    if args.ddlaFiles:
        ddla = None
        if all(['.csv' in fileName for fileName in args.ddlaFiles]):
            outputname = args.outputname if args.outputname else None
            ddla = DDLA(args.ddlaFiles[0], args.ddlaFiles[1], outputname)
            ddla.differential()
        if args.ddlaTagcloud: 
            correls = ddla.outputForTagclouds()
            OG().printTagCloudData(correls, args.maxP, outputFile = ddla.outputFile[:-4]+"_tagcloud", colorScheme = args.tagcloudcolorscheme)
            ## TODO : use this output into OutcomeGetter.


    ##multiRegression Options:
    coeffs = None
    if args.multir: #add any options that output multiR results to be caught here
        if not oa: oa = OA()
        if not fg: fg = FG()
        pprint(args)
        pprint(coeffs)
        coeffs = oa.multRegressionWithFeatures(fg, args.groupfreqthresh, args.spearman, args.bonferroni, args.p_correction_method, blacklist, whitelist, args.showfeatfreqs, args.outcomeWithOutcome, interactions = args.interactions)

        if args.csv:
            outputFile = makeOutputFilename(args, fg, oa, "multiprint") ##uses outputdir
            oa.writeSignificantCoeffs4dVis(coeffs, outputFile, args.outcomefields, paramString=str(args))
        #TODO: do some sort of sorted printing?

    # Mediation Analysis
    if args.mediation:

      # Errors and warnings

      # more than one feature location flag
      if sum([args.feat_as_path_start, args.feat_as_outcome, args.feat_as_control, args.no_features]) > 1:
        print "You must specify only one of the following: --feat_as_path_start, --feat_as_outcome, --feat_as_control, --no_features"
        sys.exit()

      # default mode, catch no feature table or no outcome table
      if not (args.feat_as_path_start or args.feat_as_outcome or args.feat_as_control or args.no_features) and (not args.feattable or args.outcometable == DEF_OUTCOME_TABLE): 
        print "You must specify a feature table (-f FEAT_TABLE) and an outcome table (--outcome_table OUTCOME_TABLE)"
        sys.exit()

      if not args.no_features:
        if args.feat_as_path_start:
          if not args.feattable:# or len(args.outcomepathstarts) == 0:
            print "You must specify a feature table: -f FEAT_TABLE"
            sys.exit()
          if len(args.outcomefields) == 0 or len(args.outcomemediators) == 0:
            print "You must specify at least one mediator and outcome"
            sys.exit()
        elif args.feat_as_outcome:
          if not args.feattable:
            print "You must specify a feature table: -f FEAT_TABLE"
            sys.exit()
          if len(args.outcomepathstarts) == 0 or len(args.outcomemediators) == 0:
            print "You must specify at least one mediator and path start"
            sys.exit()
        elif args.feat_as_control:
          if not args.feattable:
            print "You must specify a feature table: -f FEAT_TABLE"
            sys.exit()
          if len(args.outcomepathstarts) == 0 or len(args.outcomemediators) == 0 or len(args.outcomefields) == 0:
            print "You must specify at least one mediator, path start and outcome"
            sys.exit()
      else:
        if args.feattable:
          print "WARNING: You specified an feature table AND the flag --no_features. This table is being ignored."
        if args.outcometable == DEF_OUTCOME_TABLE:
          print "You must specify an outcome table"
          sys.exit()
        if len(args.outcomepathstarts) == 0 or len(args.outcomemediators) == 0 or len(args.outcomefields) == 0:
            print "You must specify at least one mediator, path start and outcome"
            sys.exit()

      path_starts = args.outcomepathstarts
      mediators = args.outcomemediators
      outcomes = args.outcomefields

      if args.outputname == '': args.outputname = None
      if args.mediationboot:
        mediation_method = "boot"
        med_boot_number = args.mediationbootnum
      else:
        mediation_method = "parametric"
        med_boot_number = 1000

      if args.feat_as_path_start:
        args.outcomefields = args.outcomemediators + args.outcomefields
        fg = FG()
        og = OG()
        MediationAnalysis(fg, og, path_starts, mediators, outcomes, args.outcomecontrols, summary=args.mediationsummary, 
              to_csv=args.mediationcsv, to_mysql=args.mediationmysql, output_name=args.outputname, 
              method=mediation_method, boot_number=med_boot_number, sig_level=args.mediationsigthresh).mediate(args.groupfreqthresh, 
              "feat_as_path_start", args.spearman, args.bonferroni,
              args.p_correction_method, logisticReg=args.logisticReg)
      elif args.feat_as_outcome:
        args.outcomefields = args.outcomepathstarts + args.outcomemediators
        fg = FG()
        og = OG()
        MediationAnalysis(fg, og, path_starts, mediators, outcomes, args.outcomecontrols, summary=args.mediationsummary, 
              to_csv=args.mediationcsv, to_mysql=args.mediationmysql, output_name=args.outputname, 
              method=mediation_method, boot_number=med_boot_number, sig_level=args.mediationsigthresh).mediate(args.groupfreqthresh, 
              "feat_as_outcome", args.spearman, args.bonferroni,
              args.p_correction_method, logisticReg=args.logisticReg)
      elif args.feat_as_control:
        controls = args.outcomecontrols
        args.outcomecontrols = []
        args.outcomefields = args.outcomepathstarts + args.outcomemediators + args.outcomefields
        fg = FG()
        og = OG()
        MediationAnalysis(fg, og, path_starts, mediators, outcomes, controls, summary=args.mediationsummary, 
              to_csv=args.mediationcsv, to_mysql=args.mediationmysql, output_name=args.outputname, 
              method=mediation_method, boot_number=med_boot_number, sig_level=args.mediationsigthresh).mediate(args.groupfreqthresh, 
              "feat_as_control", args.spearman, args.bonferroni,
              args.p_correction_method, logisticReg=args.logisticReg)
      elif args.no_features:
        args.outcomefields = args.outcomepathstarts + args.outcomemediators + args.outcomefields
        og = OG()
        MediationAnalysis(None, og, path_starts, mediators, outcomes, args.outcomecontrols, summary=args.mediationsummary, 
              to_csv=args.mediationcsv, to_mysql=args.mediationmysql, output_name=args.outputname, 
              method=mediation_method, boot_number=med_boot_number, sig_level=args.mediationsigthresh).mediate(args.groupfreqthresh, 
              "no_features", args.spearman, args.bonferroni,
              args.p_correction_method, logisticReg=args.logisticReg)
      else:
        args.outcomefields = args.outcomepathstarts + args.outcomefields
        fg = FG()
        og = OG()
        MediationAnalysis(fg, og, path_starts, mediators, outcomes, args.outcomecontrols, summary=args.mediationsummary, 
              to_csv=args.mediationcsv, to_mysql=args.mediationmysql, output_name=args.outputname, 
              method=mediation_method, boot_number=med_boot_number, sig_level=args.mediationsigthresh).mediate(args.groupfreqthresh, 
              "default", args.spearman, args.bonferroni,
              args.p_correction_method, logisticReg=args.logisticReg)


    ##Prediction methods:
    rp = None #regression predictor
    crp = None #combined regression predictor
    fgs = None #feature getters
    dr = None #Dimension Reducer

    if args.trainregression or args.testregression or args.combotestregression or args.predictregression or args.predictrtofeats or args.predictalltofeats or args.regrToLex or args.predictRtoOutcomeTable:
        if not og: og = OG()
        if not fgs: fgs = FGs()
        rp = RegressionPredictor(og, fgs, args.model)
    if args.testcombregression:
        if not og: og = OG()
        if not fgs: fgs = FGs() #all feature getters
        crp = CombinedRegressionPredictor(og, fgs, args.combmodels, args.model)
    if args.fitreducer or args.reducertolexicon:
        if not og: og = OG()
        if not fg: fg = FG()
        dr = DimensionReducer(fg, args.model, og)        
        
    if args.loadmodels and rp:
        rp.load(args.picklefile)
    
    if (args.regrToLex or args.classToLex) and isinstance(args.feattable, list):
        print "Multiple feature tables are not handled with option --prediction_to_lexicon"
        exit(1)
    elif (args.regrToLex or args.classToLex) and '16to' in args.feattable and '16to16' not in args.feattable:
        print "WARNING: using an non 16to16 feature table"
        
    if args.trainregression:
        rp.train(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse,  standardize = args.standardize)

    if args.testregression:
        rp.test(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, blacklist = blacklist,  standardize = args.standardize)

    comboScores = None
    if args.combotestregression or args.controladjustreg:
        if not og: og = OG()
        if not fg: fg = FG()
        if not rp: rp = RegressionPredictor(og, fgs, args.model)
            
        comboScores = None
        if args.combotestregression:
            comboScores = rp.testControlCombos(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, blacklist = blacklist, 
                                           noLang=args.nolang, allControlsOnly = args.allcontrolsonly, comboSizes = args.controlcombosizes, 
                                           nFolds = args.folds, savePredictions = args.pred_csv, weightedEvalOutcome = args.weightedeval,
                                           standardize = args.standardize, residualizedControls = args.res_controls)
        elif args.controladjustreg:
            comboScores = rp.adjustOutcomesFromControls(groupFreqThresh = args.groupfreqthresh, standardize = args.standardize, sparse = args.sparse, 
                                                        allControlsOnly = args.allcontrolsonly, comboSizes = args.controlcombosizes, 
                                                        nFolds = args.folds, savePredictions = args.pred_csv)
        if args.pred_csv:
            outputStream = sys.stdout
            if args.outputname:
                outputStream = open(args.outputname+'.predicted_data.csv', 'w')
            RegressionPredictor.printComboControlPredictionsToCSV(comboScores, outputStream, paramString=str(args), delimiter='|')
            print "Wrote to: %s" % str(outputStream)
            outputStream.close()
        #TODO:
        # if args.pred_feat:
        #     RegressionPredictor.printComboControlPredictionsToFeats(comboScores, label=pred_feat, paramString=str(args), delimiter='|')
        if args.csv:
            outputStream = sys.stdout
            if args.outputname:
                outputStream = open(args.outputname+'.variance_data.csv', 'w')
            RegressionPredictor.printComboControlScoresToCSV(comboScores, outputStream, paramString=str(args), delimiter='|')
            print "Wrote to: %s" % str(outputStream)
            outputStream.close()
        elif not args.pred_csv:
            pprint(comboScores)


    if args.testcombregression:
        crp.test(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, standardize = args.standardize)

    if args.predictregression:
        rp.predict(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, standardize = args.standardize)

    if args.predictrtofeats and rp:
        if not fe: fe = FE()
        rp.predictToFeatureTable(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, fe = fe, name = args.predictrtofeats, standardize = args.standardize)
    
    if args.predictRtoOutcomeTable:
        if not fgs: fgs = FGs()
        if not fe: fe = FE()
        rp.predictToOutcomeTable(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, fe = fe, name = args.predictRtoOutcomeTable)

    if args.predictalltofeats and rp:
        if not fe: fe = FE()
        rp.predictAllToFeatureTable(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, fe = fe, name = args.predictalltofeats, \
                                        standardize = args.standardize, nFolds = args.folds)
        
    if args.savemodels and rp:
        rp.save(args.picklefile)

    ##CLASSIFICATION:
    cp = None
    if args.trainclassifiers or args.testclassifiers or args.combotestclassifiers or args.predictclassifiers or args.predictctofeats or args.classToLex or args.roc or args.predictCtoOutcomeTable:
        if args.model == DEF_MODEL:#if model wasnt changed form a regression model
            args.model = DEF_CLASS_MODEL
        if not og: og = OG()
        if not fgs: fgs = FGs()
        cp = ClassifyPredictor(og, fgs, args.model) #todo change to a method variables (like og...etc..)


    if args.loadmodels and cp:
        cp.load(args.picklefile)

    if args.trainclassifiers:
        cp.train(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, standardize = args.standardize)

    if args.testclassifiers:
        cp.test(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, standardize = args.standardize)

    comboScores = None
    if args.combotestclassifiers:
        comboScores = cp.testControlCombos(groupFreqThresh = args.groupfreqthresh, standardize = args.standardize, sparse = args.sparse, blacklist = blacklist, 
                                           noLang=args.nolang, allControlsOnly = args.allcontrolsonly, comboSizes = args.controlcombosizes, 
                                           nFolds = args.folds, savePredictions = args.pred_csv, weightedEvalOutcome = args.weightedeval)
        if args.csv:
            outputStream = sys.stdout
            if args.outputname:
                outputStream = open(args.outputname+'.variance_data.csv', 'w')
            ClassifyPredictor.printComboControlScoresToCSV(comboScores, outputStream, paramString=str(args), delimiter='|')
            print "Wrote to: %s" % str(outputStream)
            outputStream.close()
        else:
            pprint(comboScores)
        if args.pred_csv:
            outputStream = sys.stdout
            if args.outputname:
                outputStream = open(args.outputname+'.predicted_data.csv', 'w')
            ClassifyPredictor.printComboControlPredictionsToCSV(comboScores, outputStream, paramString=str(args), delimiter='|')
            print "Wrote to: %s" % str(outputStream)
            outputStream.close()

    if args.predictclassifiers:
        cp.predict(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse)

    if args.roc:
        cp.roc(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, output_name = args.outputname if args.outputname else "ROC",  standardize = args.standardize)

    if args.predictctofeats and cp:
        if not fe: fe = FE()
        cp.predictToFeatureTable(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, fe = fe, name = args.predictctofeats)

    if args.predictCtoOutcomeTable:
        if not fgs: fgs = FGs()
        if not fe: fe = FE()
        cp.predictToOutcomeTable(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, fe = fe, name = args.predictCtoOutcomeTable)

    if args.predictalltofeats and cp:
        if not fe: fe = FE()
        cp.predictAllToFeatureTable(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, fe = fe, name = args.predictalltofeats, \
                                        standardize = args.standardize, nFolds = args.folds)

    c2rp = None
    if args.trainclasstoreg or args.testclasstoreg or args.predictclasstoreg:
        if not og: og = OG()
        if not fg: fg = FG()
        c2rp = ClassifyToRegressionPredictor(og, fg, modelR = args.model) #todo change to a method variables (like og...etc..)

    if args.trainclasstoreg:
        c2rp.train(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse)

    if args.testclasstoreg:
        c2rp.test(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse)

    if args.predictclasstoreg:
        c2rp.predict(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse)

    if args.savemodels and cp:
        cp.save(args.picklefile)

    if args.loadmodels and dr:
        dr.load(args.picklefile)

    if args.classToLex or args.regrToLex:
        lexicon_dict = None
        if rp and not cp:
            print "----- Detected a regressor"
            lexicon_dict = rp.getWeightsForFeaturesAsADict()  #returns featTable -> category -> term -> weight
        elif cp:
            print "----- Detected a classifier"
            lexicon_dict = cp.getWeightsForFeaturesAsADict()  #returns featTable -> category -> term -> weight
        
        lex_dict_with_name = {args.classToLex: v for featTableName,v in lexicon_dict.iteritems()} if args.classToLex else {args.regrToLex: v for featTableName,v in lexicon_dict.iteritems()}
        # print lex_dict_with_name.items()
        for lexName, lexicon in lex_dict_with_name.iteritems():
            lex = lexInterface.WeightedLexicon(lexicon, mysql_host = args.mysql_host)
            lex.createWeightedLexiconTable('dd_'+lexName)

    if args.fitreducer:
        #dr.fit(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse, blacklist = blacklist)
        dr.fit(groupFreqThresh = args.groupfreqthresh, sparse = args.sparse)
    
    if args.reducertolexicon:
        lexicons = dr.modelToLexicon()
        for outcomeName, lexDict in lexicons.iteritems():
            lexiconName = args.reducertolexicon
            if outcomeName != 'noOutcome':
                lexiconName += '_'+outcomeName
            lexicon = lexInterface.WeightedLexicon(lexDict, mysql_host = args.mysql_host)
            lexicon.createLexiconTable(lexiconName)
        
    if args.savemodels and dr:
        dr.save(args.picklefile)

    ##Plot Actions:
    if args.barplot:
        outputFile = makeOutputFilename(args, fg, oa, "barplot")
        oa.barPlot(correls, outputFile)

    if args.descplot:
        if not og: og=OG()
        (groups, outcome_to_gid_to_value, controls) = og.getGroupsAndOutcomes(args.groupfreqthresh)
        outcome_to_values = dict(  map(lambda (k,v): (k, v.values()), outcome_to_gid_to_value.items())  )
        outputFile = makeOutputFilename(args, None, og, "desc_stats")
        from FeatureWorker.lib.descStats import StatsPlotter
        sp = StatsPlotter(args.corpdb)
        sp.plotDescStats(outcome_to_values, len(groups), outputFile)

    if args.scatterplot: # or args.hist2d:
        scatter_dict_1 = {}
        scatter_dict_2 = {}

        if not og: og = OG()
        (out_groups, outcome_to_gid_to_value, controls) = og.getGroupsAndOutcomes(args.groupfreqthresh)
        scatter_dict_1 = outcome_to_gid_to_value

        if args.featnames and args.feattable:
            if not fg: fg = FG()
            where = 'group_id IN ( %s )'%(','.join(map(str, out_groups)))
            (feat_groups, feature_to_gid_to_value) = fg.getGroupsAndFeats( where )
            scatter_dict_2 = feature_to_gid_to_value
        else:
            scatter_dict_2 = outcome_to_gid_to_value
            
        from FeatureWorker.lib.descStats import StatsPlotter
        sp = StatsPlotter()
        for scatter_group_1 in scatter_dict_1:
            for scatter_group_2 in scatter_dict_2:
                if scatter_group_1 != scatter_group_2 or (args.featnames and args.feattable):
                    (xValues, yValues) = alignDictsAsLists(scatter_dict_1[scatter_group_1], scatter_dict_2[scatter_group_2])
                    if args.scatterplot:
                        outputFile = '%s/plots/scatter/%s.%s.%s.%s.scat'%(args.outputdir, fg.featureTable if fg else 'noFeatureTable', scatter_group_2, og.outcome_table, scatter_group_1)
                        if args.outputname: outputFile = '.'.join([args.outputname, scatter_group_1, scatter_group_2, 'scat'])
                        sp.plotScatter(scatter_group_1, xValues, scatter_group_2, yValues, outputFile)
                    if args.hist2d:
                        outputFile = '%s/plots/scatter/%s.%s.%s.%s.hist2d'%(args.outputdir, fg.featureTable if fg else 'noFeatureTable', scatter_group_2, og.outcome_table, scatter_group_1)
                        if args.outputname: outputFile = outputFile = '.'.join([args.outputname, scatter_group_1, scatter_group_2, 'hist2d'])
                        sp.plot2dHist(scatter_group_1, xValues, scatter_group_2, yValues, outputFile)
                    outputFile = '%s/plots/scatter/%s.%s.%s.%s.desc.%s'%(args.outputdir, fg.featureTable if fg else 'noFeatureTable', scatter_group_2, og.outcome_table, scatter_group_1, scatter_group_1)
                    if args.outputname: outputFile = '.'.join([args.outputname, scatter_group_1, scatter_group_2, 'desc'])
                    sp.plotDescStats(dict({scatter_group_1:xValues}), outputFile)
                    outputFile = '%s/plots/scatter/%s.%s.%s.%s.desc.%s'%(args.outputdir, fg.featureTable if fg else 'noFeatureTable', scatter_group_2, og.outcome_table, scatter_group_1, scatter_group_2)
                    if args.outputname: outputFile = '.'.join([args.outputname, scatter_group_1, scatter_group_2, 'desc'])
                    sp.plotDescStats(dict({scatter_group_2:yValues}), outputFile)
    
    #Analysis using only feature tables:
    if args.ttestfeats:
        if not fgs: fgs = FGs()
        #TODO: change below to use every pair of FGs (in case there are more than 2)
        results = fgs[0].ttestWithOtherFG(fgs[1], maskTable = args.masktable, groupFreqThresh=args.groupfreqthresh)
        if args.csv:
            #add this
            #can use something similar to OG correlMatrix
            pass
        pprint(results)

    if args.notify or args.notifyluke or args.notifyandy or args.notifyjohannes:
        from FeatureWorker.lib import notify
        if args.notify:
            notify.sendEmail("featureWorker run Finished", "this was sent from featureWorker.py", args.notify)
        if args.notifyluke: 
            notify.sendEmail("featureWorker run Finished", args.notifyluke + '\n\n\n' + str(args), 'lukaszad@gmail.com')
        if args.notifyandy: 
            notify.sendEmail("featureWorker run Finished", args.notifyandy + '\n\n\n' + str(args), 'andy.schwartz@gmail.com')
        if args.notifyjohannes: 
            notify.sendEmail("featureWorker run Finished", args.notifyjohannes + '\n\n\n' + str(args), 'johannes.chicago@gmail.com')  

    if args.toinitfile:
      with open(args.toinitfile, 'w') as init_file:  
        init_file.write("[constants]\n")
        
        if (args.corpdb and args.corpdb != DEF_CORPDB): init_file.write("corpdb = " + str(args.corpdb)+"\n") 
        if (args.corptable and args.corptable != DEF_CORPTABLE): init_file.write("corptable = " + str(args.corptable)+"\n") 
        if (args.correl_field and args.correl_field != DEF_CORREL_FIELD): init_file.write("correl_field = " + str(args.correl_field)+"\n") 
        if (args.mysql_host and args.mysql_host != "localhost"): init_file.write("mysql_host = " + str(args.mysql_host)+"\n") 
        if (args.message_field and args.message_field != DEF_MESSAGE_FIELD): init_file.write("message_field = " + str(args.message_field)+"\n") 
        if (args.messageid_field and args.messageid_field != DEF_MESSAGEID_FIELD): init_file.write("messageid_field = " + str(args.messageid_field)+"\n") 
        if (args.lexicondb and args.lexicondb != DEF_LEXICON_DB): init_file.write("lexicondb = " + str(args.lexicondb)+"\n") 
        if (args.feattable and args.feattable != DEF_FEAT_TABLE): init_file.write("feattable = " + str(args.feattable)+"\n") 
        if (args.featnames and args.featnames != DEF_FEAT_NAMES): init_file.write("featnames = " + ", ".join([str(feat) for feat in args.featnames])+"\n") 
        if (args.date_field and args.date_field != DEF_DATE_FIELD): init_file.write("date_field = " + str(args.date_field)+"\n")
        if (args.outcometable and args.outcometable != DEF_OUTCOME_TABLE): init_file.write("outcometable = " + str(args.outcometable)+"\n") 
        if (args.outcomefields and args.outcomefields != DEF_OUTCOME_FIELDS): init_file.write("outcomefields = " + ", ".join([str(out) for out in args.outcomefields])+"\n")
        if (args.outcomecontrols and args.outcomecontrols != DEF_OUTCOME_CONTROLS): init_file.write("outcomecontrols = " + ", ".join([str(out) for out in args.outcomecontrols])+"\n")
        if (args.outcomeinteraction and args.outcomeinteraction != DEF_OUTCOME_CONTROLS): init_file.write("outcomeinteraction = " + ", ".join([str(out) for out in args.outcomeinteraction])+"\n")
        if (args.featlabelmaptable and args.featlabelmaptable != ''): init_file.write("featlabelmaptable = " + str(args.featlabelmaptable)+"\n")
        if (args.featlabelmaplex and args.featlabelmaplex != ''): init_file.write("featlabelmaplex = " + str(args.featlabelmaplex)+"\n")
        if (args.wordTable): init_file.write("wordTable = " + str(args.wordTable)+"\n")
        if (args.outputname): init_file.write("outputname = " + str(args.outputname)+"\n")
        if (args.groupfreqthresh and args.groupfreqthresh != int(DEF_GROUP_FREQ_THRESHOLD)): init_file.write("groupfreqthresh = " + str(args.groupfreqthresh)+"\n")
        if (args.lextable): init_file.write("lextable = " + str(args.lextable)+"\n")
        if (args.p_correction_method): init_file.write("p_correction_method = " + str(args.p_correction_method)+"\n")
        if (args.tagcloudcolorscheme and args.tagcloudcolorscheme != 'multi'): init_file.write("tagcloudcolorscheme = " + str(args.tagcloudcolorscheme)+"\n")
        if (args.maxP and args.maxP != float(DEF_P)): init_file.write("maxP = " + str(args.maxP)+"\n")
        
        init_file.close()                

    _warn("--\nInterface Runtime: %.2f seconds"% float(time.time() - start_time))
    _warn("featureWorker exits with success! A good day indeed :).")

if __name__ == "__main__":
    main()
    sys.exit(0)
        
