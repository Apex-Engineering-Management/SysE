# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#import syse

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../syse/'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SysE'
copyright = '2023, Apexpromgt'
author = 'Jonathon Nicholson'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.imgmath',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'jupyter_sphinx',
    'hoverxref.extension',
    'sphinx_charts.charts'
]

templates_path = ['_templates']
exclude_patterns = []

# autosummary_generate = True
# autoclass_content = 'both'
# html_show_sourcelink = False
# autodoc_inherit_docstrings = True
# set_type_checking_flag = True
# autodoc_default_flags = ['members']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "_static/apex-logo.svg"
html_favicon = "_static/apex-logo.svg"

# Options for ePUB output

epub_title = 'Using SysE to develop data applications'
epub_theme = 'epub'
epub_cover = ('_static/cover.png', 'epub-cover.html')

html_math_renderer = 'imgmath'
imgmath_image_format = 'png'



