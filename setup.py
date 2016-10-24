import sys
import os
import subprocess

_version = sys.version_info[0]

try:
  from setuptools import setup
except ImportError:
  if _version >= 3:
    sys.exit("Need setuptools to install dlatk for Python 3.x")
  from distutils.core import setup


DESCRIPTION = "DLATK is an end to end human text analysis package, specifically suited for social media and social scientific applications. It is written in Python 3 and developed by the World Well-Being Project at the University of Pennsylvania. "
LONG_DESCRIPTION = """
DLATK v1.0
----------

This package offers end to end text analysis: feature extraction, correlation, 
mediation and prediction / classification. For more information please visit:

  * http://dlatk.wwbp.org
  * https://www.github.com/dlatk/dlatk
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
LICENSE = 'GNU General Public License v3 (GPLv3)'
AUTHOR = "H. Andrew Schwartz, Salvatore Giorgi, Maarten Sap, Patrick Crutchley, Lukasz Dziurzynski and Megha Agrawal"
EMAIL = "hansens@sas.upenn.edu, sgiorgi@sas.upenn.edu"
MAINTAINER = "Salvatore Giorgi, H. Andrew Schwartz, Patrick Crutchley"
MAINTAINER_EMAIL = "sgiorgi@sas.upenn.edu, hansens@sas.upenn.edu, pcrutchl@psych.upenn.edu"
URL = "http://dlatk.wwbp.org"
DOWNLOAD_URL = 'https://github.com/dlatk/dlatk'
CLASSIFIERS = [
  'Environment :: Console',
  'Natural Language :: English',
  'Intended Audience :: End Users/Desktop',
  'Intended Audience :: Developers',
  'Intended Audience :: Science/Research',
  'Programming Language :: Python',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.5',
  'Topic :: Scientific/Engineering',
]
VERSION = '1.0.0'
PACKAGE_DATA = {
  'dlatk': ['data/*.sql'],
}
INCLUDE_PACKAGE_DATA = True
SETUP_REQUIRES = [
  'numpy', 
]
INSTALL_REQUIRES = [
  'matplotlib>=1.3.1', 
  'mysqlclient', 
  'nltk>=3.1', 
  'numpy', 
  'pandas>=0.17.1', 
  'patsy>=0.2.1', 
  'python-dateutil>=2.5.0', 
  'scikit-learn>=0.17.1', 
  'scipy>=0.13.3', 
  'SQLAlchemy>=0.9.9', 
  'statsmodels>=0.5.0', 
]
EXTRAS_REQUIRE = {
  'image': ['image'],
  'langid': ['langid>=1.1.4'],
  'rpy2': ['rpy2'],
  'wordcloud':  ['wordcloud>1.1.3'],
}

SCRIPTS = ['dlatkInterface.py']

if __name__ == "__main__":

  setup(name=DISTNAME,
      author=AUTHOR,
      author_email=EMAIL, 
      version=VERSION,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
      include_package_data=INCLUDE_PACKAGE_DATA,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      classifiers=CLASSIFIERS,
      setup_requires=SETUP_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      install_requires=INSTALL_REQUIRES,
      scripts = SCRIPTS,
  )

