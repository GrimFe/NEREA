# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  

project = 'NEREA'
copyright = 'YYYY-%Y, Federico Grimaldi'
author = 'Federico Grimaldi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo",
              "sphinx.ext.viewcode",
              "sphinx.ext.autodoc",
              "myst_parser",
              "sphinx.ext.autosummary",
              "sphinx.ext.napoleon"]

todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'undoc-members': False,       # Don’t include undocumented stuff
    'private-members': False,     # Don’t include _private attributes
    'special-members': False,     # Don’t include __init__, __repr__, etc.
    'inherited-members': False,   # Skip inherited methods unless needed
    'show-inheritance': True,     # Optional: shows base class in docs
    'exclude-members': '__weakref__',  # Clean up unwanted auto-generated members
}
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


myst_enable_extensions = [
    "colon_fence",     # allows ::: fenced blocks
    "deflist",         # definition lists
    "html_image",      # renders <img> and Markdown images
]

html_static_path = ["_static", "../../img"]
