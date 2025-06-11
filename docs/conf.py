import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google/Numpy style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",   # Optional: better type hinting
]

html_theme = "sphinx_rtd_theme"
