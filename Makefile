# Makefile to reproduce the results of the WSDM2019 paper
# "When Users Change Their Mind: Off-Policy Evaluation in Non-stationary
# Environments"

BUILD  ?= build
OUT    ?= out
PYTHON ?= python

include scripts/data.mk
include scripts/train.mk
include scripts/estimators.mk

all: lastfm/results delicious/results

