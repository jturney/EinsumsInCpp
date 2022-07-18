import os
import re
import sys
import importlib

# Minimum version, enforced by sphinx
needs_sphinx = '4.3'

# -----------------------------------------------------------------------
# General configuration
# -------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

sys.path.insert(0, os.path.abspath('../sphinxext'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.mathjax',
    'sphinx_panels',
]

skippable_extensions = [
    ('breathe', 'skip generating C/C++ API from comment blocks.'),
]

for ext, warn in skippable_extensions:
    ext_exist = importlib.util.find_spec(ext) is not None
    if ext_exist:
        extensions.append(ext)
    else:
        print(f"Unable to find Sphinx extension '{ext}', {warn}.")

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# General substitutions
project = 'Einsums'
copyright = '2022, Einsums Developers'

version = "0.1"
release = "0.1"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

def setup(app):
    # add a config value for `ifconfig` directives
    app.add_config_value('python_version_major', str(sys.version_info.major), 'env')

# -----------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_favicon = '_static/favicon/favicon.ico'

# Set up the version switcher.  The versions.json is stored in the doc repo.
if os.environ.get('CIRCLE_JOB', False) and \
        os.environ.get('CIRCLE_BRANCH', '') != 'main':
    # For PR, name is set to its ref
    switcher_version = os.environ['CIRCLE_BRANCH']
elif ".dev" in version:
    switcher_version = "devdocs"
else:
    switcher_version = f"{version}"

html_theme_options = {
  "logo": {
      "image_light": "einsumslogo.svg",
      "image_dark": "einsumslogo_dark.svg",
  },
  "github_url": "https://github.com/jturney/EinsumsInCpp",
  "collapse_navigation": True,
#   "external_links": [
    #   {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"}
    #   ],
  # Add light/dark mode and documentation version switcher:
  "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
  "switcher": {
      "version_match": switcher_version,
      "json_url": "_static/versions.json",
  },
}

html_title = "%s v%s Manual" % (project, version)
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
html_css_files = ["einsums.css"]
html_context = {"default_mode": "light"}

# Prevent sphinx-panels from loading bootstrap css, the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

htmlhelp_basename = 'einsums'

if 'sphinx.ext.pngmath' in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']

mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"

plot_html_show_formats = False
plot_html_show_source_link = False

# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# XeLaTeX for better support of unicode characters
latex_engine = 'xelatex'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
_stdauthor = 'Written by the Einsums community'
latex_documents = [
  ('reference/index', 'einsums-ref.tex', 'Einsums Reference',
   _stdauthor, 'manual'),
  ('user/index', 'einsums-user.tex', 'Einsums User Guide',
   _stdauthor, 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

latex_elements = {
    'fontenc': r'\usepackage[LGR,T1]{fontenc}'
}

# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r'''
% In the parameters section, place a newline after the Parameters
% header
\usepackage{xcolor}
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% but expdlist old LaTeX package requires fixes:
% 1) remove extra space
\usepackage{etoolbox}
\makeatletter
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
\makeatother
% 2) fix bug in expdlist's way of breaking the line after long item label
\makeatletter
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % now a hack because Sphinx inserts \leavevmode after term node
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
\makeatother
% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother
% Fix footer/header
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
'''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = False


# -----------------------------------------------------------------------------
# Texinfo output
# -----------------------------------------------------------------------------

texinfo_documents = [
  ("contents", 'einsums', 'Einsums Documentation', _stdauthor, 'Einsums',
   "Einsums: tensor library.",
   'Programming',
   1),
]


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}


# -----------------------------------------------------------------------
# Einsums extensions
# -----------------------------------------------------------------------

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = 'dump.xml'

# -----------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------

autosummary_generate = True

# -----------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [('png', 100), 'pdf']

import math
phi = (math.sqrt(5) + 1)/2

plot_rcparams = {
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}


# -----------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------

# import inspect
# from os.path import relpath, dirname

# for name in ['sphinx.ext.linkcode']:
#     try:
#         __import__(name)
#         extensions.append(name)
#         break
#     except ImportError:
#         pass
# else:
#     print("NOTE: linkcode extension not found -- no links to source generated")

# -----------------------------------------------------------------------
# Breathe & Doxygen
# -----------------------------------------------------------------------
breathe_projects = dict(einsums=os.path.join("..", "build", "doxygen", "xml"))
breathe_default_project = "einsums"
breathe_default_members = ("members", "undoc-members", "protected-members")
