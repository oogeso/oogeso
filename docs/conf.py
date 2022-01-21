# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "Oogeso"
copyright = "2022, SINTEF"
author = "SINTEF"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.viewcode",
    # "sphinxcontrib.bibtex",
    # "sphinx_panels",
    # "sphinxext.rediraffe",
    # "sphinxcontrib.mermaid",
    # "sphinxext.opengraph",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/logo_oogeso.png"
html_favicon = "_static/logo_oogeso.png"
html_static_path = "_static/"
html_title = "Oogeso Docs"
html_theme_options = {
    "github_url": "https://github.com/oogeso/oogeso",
    "repository_url": "https://github.com/oogeso/oogeso",
    "use_edit_page_button": True,
    "repository_branch": "master",
    "path_to_docs": "docs.yml",
}
# OpenGraph metadata
# ogp_site_url = "https://myst-parser.readthedocs.io/en/latest"
# This is the image that GitHub stores for our social media previews
# ogp_image = "https://repository-images.githubusercontent.com/240151150/316bc480-cc23-11eb-96fc-4ab2f981a65d"  # noqa: E501
# ogp_custom_meta_tags = ['<meta name="twitter:card" content="summary_large_image">',]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_number_code_blocks = ["typescript"]
myst_heading_anchors = 2
myst_footnote_transition = True
myst_dmath_double_inline = True
panels_add_bootstrap_css = True

suppress_warnings = ["myst.strikethrough"]
