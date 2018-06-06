"""HumanTree: Count trees on a property."""

import os

from . import humantree  # noqa: F401

dir_name = os.path.dirname(os.path.realpath(__file__))

authors = []
contributors = []

with open(os.path.join(dir_name, 'contributors.txt')) as f:
    for cont in f.read().splitlines():
        if '*' in cont:
            authors.append(cont.split('(')[0].strip(' *'))
        else:
            contributors.append(cont.split('(')[0].strip())

__version__ = '0.1.0'
__author__ = ' & '.join([', '.join(authors[:-1]), authors[-1]])
__contributors__ = (' & '.join([', '.join(
    contributors[:-1]), contributors[-1]])) if len(contributors) else ''
__license__ = 'MIT'
