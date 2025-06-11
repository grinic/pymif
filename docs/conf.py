import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Example'
copyright = 'workshop participant'
author = 'workshop participant'
release = '0.1'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google/Numpy style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",   # Optional: better type hinting
    'myst_parser', 
    'autoapi.extension', 
    'sphinx.ext.linkcode'
]
autoapi_dirs = ['../pymif']

def linkcode_resolve(domain, info):
    
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/grinic/pymif/blob/main/%s.py" % filename

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
