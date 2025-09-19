import sys
import tomllib
from datetime import datetime

sys.path.insert(0, "..")

with open("../pyproject.toml", "rb") as f:
    toml_data = tomllib.load(f)["project"]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# noinspection PyUnresolvedReferences
project = toml_data["name"]
# noinspection PyShadowingBuiltins
copyright = f"2014-{datetime.now().year}, xrscipy Developers"
# noinspection PyUnresolvedReferences
author = ", ".join(f"{e['name']} <{e['email']}>" for e in toml_data["authors"])
# noinspection PyUnresolvedReferences
release = toml_data["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_parser",
]

extlinks = {
    "issue": ("https://github.com/hippalectryon-0/xr-scipy/issues/%s", "GH"),
    "pull": ("https://github.com/hippalectryon-0/xr-scipy/pull/%s", "PR"),
    "doi": ("https://dx.doi.org/%s", "doi:%s"),
}

autosummary_generate = True
numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

# From old setup.py
html_last_updated_fmt = "%Y-%m-%d"
htmlhelp_basename = "xrscipydoc"
latex_documents = [
    ("index", "xrscipy.tex", "xr-scipy Documentation", "xr-scipy Developers", "manual"),
]
man_pages = [("index", "xrscipy", "xr-scipy Documentation", ["xr-scipy Developers"], 1)]

texinfo_documents = [
    (
        "index",
        "xrscipy",
        "xr-scipy Documentation",
        "xr-scipy Developers",
        "xr-scipy",
        "Scipy for xarray",
        "Miscellaneous",
    ),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
