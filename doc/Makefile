# Makefile for Sphinx documentation

PYVER:=$(shell python3 -c 'from sys import version_info as v; print("{0}.{1}".format(v[0], v[1]))')
PYTHON = python$(PYVER)

# You can set these variables from the command line
SPHINXOPTS  ?=
SPHINXBUILD ?= LANG=C sphinx-build
PAPER       ?=
DOXYGEN     ?= doxygen

# For merging a documentation archive into a git checkout of einsums/doc
# Turn a tag like v1.18.0 into 1.18
# Use sed -n -e 's/patttern/match/p' to return a blank value if no match
TAG ?= $(shell git describe --tag | sed -n -e's,v\([1-9]\.[0-9]*\)\.[0-9].*,\1,p')

FILES=

# Internal variables
PAPEROPT_a4 = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS = -WT --keep-going -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html

#---------------------------------------------------------------------

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean     to remove generated doc files and start fresh"
	@echo "  html      to make standalone HTML files"

clean:
	-rm -rf build/*
	find . -name generated -type d -prune -exec rm -f "{}" ";"

#------------------------------------------------------------------------
# Automated generation of all documents
#------------------------------------------------------------------------

GITVER ?= $(shell cd ..; $(PYTHON) -c "import versioneer as v; print(v.get_versions()['full-revisionid'][:10])")

#------------------------------------------------------------------------------
# Basic Sphinx generation rules for different formats
#------------------------------------------------------------------------------

generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst)
	mkdir -p build
	touch build/generate-stamp

html: html-build
html-build: generate
	mkdir -p build/html build/doctrees
	$(PYTHON) preprocess.py
ifeq (, $(shell which $(DOXYGEN)))
	@echo "Unable to find 'Doxygen:$(DOXYGEN)', skip generating C/C++ API from comment blocks."
else
	$(DOXYGEN) build/doxygen/Doxyfile
endif
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html $(FILES)
	$(PYTHON) postprocess.py html build/html/*.html
	@echo
	@echo "Build finished. The HTML pages are in build/html."
