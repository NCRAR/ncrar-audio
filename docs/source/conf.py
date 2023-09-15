# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NCRAR audio'
copyright = '2023, NCRAR audio development team'
author = 'NCRAR audio development team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxcontrib.apidoc',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
    #'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []


apidoc_module_dir = '../../ncrar_audio'
apidoc_output_dir = 'api'
apidoc_separate_modules = True

autodoc_mock_imports = ['pyxid2']
autosummary_generate = True
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
