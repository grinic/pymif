# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyMIF'
copyright = '2025, Nicola Gritti'
author = 'Nicola Gritti'
release = 'v0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",        # Google/Numpy style docstrings
    # "sphinx.ext.viewcode",
    # "sphinx_autodoc_typehints",   # Optional: better type hinting
    'myst_parser', 
    'autoapi.extension', 
    'sphinx.ext.linkcode'
]

autoapi_dirs = ['../../pymif']

def linkcode_resolve(domain, info):
    
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/grinic/pymif/blob/main/%s.py" % filename

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
