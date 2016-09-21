import sys
import subprocess

_version = sys.version_info[0]

try:
  from setuptools import setup
except ImportError:
  if _version >= 3:
    sys.exit("Need setuptools to install dlatk for Python 3.x")
  from distutils.core import setup

def _get_input(question):
  return input(question) if _version >= 3 else raw_input(question)

# check to see if mysql is running
def check_mysql():
  msqlr = subprocess.Popen("ps aux".split(), stdout=subprocess.PIPE).stdout
  grep = subprocess.Popen(["grep", "mysql"], stdin=msqlr, stdout=subprocess.PIPE, universal_newlines=True).stdout
  msqlrLines = grep.read().split("\n")
  if len(msqlrLines) == 1 and "grep" in msqlrLines[0]:
    sys.exit("MySQL is not running. Please restart or install MySQL to continue.")
  return True

def get_user_config():
  import MySQLdb
  db = MySQLdb.connect(host="localhost",user="root",passwd="****")
  return db.cursor()

def install_table(db_name, table_name, cursor):
  sql = 'CREATE table %s.%s' % (db_name, table_name)
  cursor.execute(sql)
  return

def install_db(db_name, cursor):
  sql = 'CREATE DATABASE %s' % (db_name)
  cursor.execute(sql)

  return

def install_data(data, cursor):
  tables, database = data.popitem()
  install_db(database, cursor)
  for table in tables:
    print("Installing %s.%s" % (database, table))
    install_table(database, table, cursor)


DESCRIPTION = "DLATK is an end to end text analysis package developed by the World Well-Being Project at the University of Pennsylvania."
LONG_DESCRIPTION = """
DLATK v1.0
----------

This package offers end to end text analysis: feature extraction, correlation, 
mediation and prediction / classification. For more information please visit:

  * http://dlatk.wwbp.org
  * https://www.github.com/wwbp/dlatk
  * http://wwbp.org

CONTACT
-------

Please send bug reports, patches, and other feedback to

  Salvatore Giorgi (sgiorgi@sas.upenn.edu) or H. Andrew Schwartz (hansens@sas.upenn.edu)

"""
DISTNAME = 'dlatk'
PACKAGES = ['dlatk',
  'dlatk.lib',
  'dlatk.LexicaInterface',
  'dlatk.mysqlMethods',
]
LICENSE = 'Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States License'
AUTHOR = "H. Andrew Schwartz, Salvatore Giorgi, Maarten Sap, Patrick Crutchley, Lukasz Dziurzynski and Megha Agrawal"
EMAIL = "hansens@sas.upenn.edu, sgiorgi@sas.upenn.edu"
MAINTAINER = "Salvatore Giorgi, H. Andrew Schwartz, Patrick Crutchley"
MAINTAINER_EMAIL = "sgiorgi@sas.upenn.edu, hansens@sas.upenn.edu, pcrutchl@psych.upenn.edu"
URL = "http://dlatk.wwbp.org"
DOWNLOAD_URL = 'https://github.com/wwbp/dlatk'
CLASSIFIERS = [
  'Environment :: Console',
  'Natural Language :: English',
  'Intended Audience :: End Users/Desktop',
  'Intended Audience :: Developers',
  'Intended Audience :: Science/Research',
  'Programming Language :: Python',
  'Programming Language :: Python :: 2',
  'Programming Language :: Python :: 2.7',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.5',
  'Topic :: Scientific/Engineering',
]
VERSION = '1.0.dev4'
PACKAGE_DATA = {'dlatk': ['data/*.sql',
            'data/*.csv',],
}
INSTALL_REQUIRES = [
  'image',
  'matplotlib>=1.3.1', 
  'mysqlclient', 
  'nltk>=3.1', 
  'numpy', 
  'pandas>=0.17.1', 
  'patsy>=0.2.1', 
  'python-dateutil>=2.5.0', 
  'rpy2',
  'scikit-learn>=0.17.1', 
  'scipy>=0.13.3', 
  'SQLAlchemy>=0.9.9', 
  'statsmodels>=0.5.0', 
  'wordcloud>1.1.3', 
]
INSTALL_DATA = {
  "tutorial": {"dla_tutorial": []},
  "lexica": {"permaLexicon": []},
}

if __name__ == "__main__":

  check_mysql()

  # setup(name=DISTNAME,
  #     author=AUTHOR,
  #     author_email=EMAIL, 
  #     maintainer=MAINTAINER,
  #     maintainer_email=MAINTAINER_EMAIL,
  #     version=VERSION,
  #     packages=PACKAGES,
  #     package_data=PACKAGE_DATA,
  #     description=DESCRIPTION,
  #     long_description=LONG_DESCRIPTION,
  #     license=LICENSE,
  #     url=URL,
  #     download_url=DOWNLOAD_URL,
  #     classifiers=CLASSIFIERS,
  #     install_requires=INSTALL_REQUIRES,
  # )
  
  mysql_user_info = None

  # install dla_tutorial
  print("DLATK comes with a small blog corpus for tutorial purposes.")
  sample_input = _get_input("Would you like to import this directly into MySQL? (y/n) ")
  
  if sample_input.lower().startswith("y"):
    mysql_user_info = get_user_config()
    install_data(INSTALL_DATA["tutorial"], mysql_user_info)
  else:
    print("Skipping tutorial data.")

  # install permaLexicon
  print("DLATK comes with a small blog corpus for tutorial purposes.")
  lex_input = _get_input("Would you like to install this now? (y/n) ")
  
  if lex_input.lower().startswith("y"):
    if not mysql_user_info:
      mysql_user_info = get_user_config()
    install_data(INSTALL_DATA["lexica"], mysql_user_info)
  else:
    print("Skipping lexica data.")




