# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime

# Add source code path
sys.path.insert(0, os.path.abspath('../../EQTransformer'))


# -- Project information -----------------------------------------------------

first_year = 2019
last_year = int(datetime.datetime.now().year)

if first_year == last_year:
    years = str(first_year)
else:
    years = "{}-{}".format(first_year, last_year)

project = 'EQTransformer'
copyright = years + ', S.Mostafa Mousavi'
author = 'S. Mostafa Mousavi


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "nbsphinx"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**tests**", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Mock these imports for lightweight docs building
autodoc_mock_imports = ["tensorflow", "numpy"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Display todos
todo_include_todos = True

# Fail process if an exception occurs while running notebooks
nbsphinx_allow_errors = False
nbsphinx_execute = "never"
nbsphinx_timeout = 1800  # 30 mins, need longer cell timeout for training models

numfig = True  # number figures with captions
