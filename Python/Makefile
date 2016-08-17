# Creates and runs a python virtualenv with jupyter and all requirements
# usage: make run

.PHONY: run test
# Defined run and test as tasks, not targets.
# they will always be executed.
 

# Define the default shell that make will use:
SHELL := /bin/bash


# if running on MAC OSx, you need to install coreutils first
# brew install coreutils to use greadlink.
# Get readlink (if running on Linux) or greadlink (if running on Mac)
READLINK := $(shell command -v greadlink || echo readlink)


# Define exact python version and path
PYTHON ?= $(shell which python2.7)
# Define exact virtualenv path
VIRTUALENV ?= $(shell which virtualenv) -p $(PYTHON)

# Target that creates virtualenv with requirements installed
devenv: Makefile requirements.txt

	# Create directory if it does not exist
	test -r devenv || $(VIRTUALENV) devenv
	
	# Activate virtualenv and install requirements
	source devenv/bin/activate ; python devenv/bin/pip install -r ./requirements.txt
	
	# Update the last modified timestamp
	touch devenv	

distclean:
	rm -rf devenv/
