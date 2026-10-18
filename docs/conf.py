import os
import sys

# Ensure the package root is in sys.path
sys.path.insert(0, os.path.abspath('..'))

project = 'probability'
copyright = '2026, Vahndi Minah'
author = 'Vahndi Minah'

# Sphinx extensions
extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# AutoAPI configuration
autoapi_type = 'python'
autoapi_dirs = ['../probability']
autoapi_root = 'api'
autoapi_keep_files = False
autoapi_add_toctree_entry = True
autoapi_python_class_content = 'both'
autoapi_member_order = 'bysource'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

# Napoleon settings for docstrings (supports both NumPy and Google formats)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_static_path = []
