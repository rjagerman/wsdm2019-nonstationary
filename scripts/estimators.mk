# Recipes for estimators
LASTFM_EST_DEPS := $(BUILD)/data/lastfm/bandit_items.gz
LASTFM_EST_DEPS += $(BUILD)/data/lastfm/bandit_users.gz
LASTFM_EST_DEPS += $(BUILD)/data/lastfm/bandit_clusters.gz
LASTFM_EST_DEPS += lastfm/logger
LASTFM_EST_DEPS += lastfm/candidates

DELICIOUS_EST_DEPS := $(BUILD)/data/delicious/bandit_items.gz
DELICIOUS_EST_DEPS += $(BUILD)/data/delicious/bandit_users.gz
DELICIOUS_EST_DEPS += $(BUILD)/data/delicious/bandit_clusters.gz
DELICIOUS_EST_DEPS += delicious/logger
DELICIOUS_EST_DEPS += delicious/candidates

# $(1) = dataset
# $(2) = stationarity type (linear, abrupt, stationariy)
# $(3) = aggregator type (avg, exp, win, adaexp, adawin)
# $(4) = seed
# $(5) = alpha
# $(6) = tau
# $(7) = factor
# $(8) = adapt
define perform_estimation
	mkdir -p $(BUILD)/results/$(1)/$(2)/$(3)/$(4).in_progress
	$(PYTHON) -m experiments.hetrec.evaluation \
		--logging $(BUILD)/models/$(1)/logger.gz \
		--policies $(BUILD)/models/$(1)/candidate_*.gz \
		--estimator ips \
		--change $(2) \
		--aggregator $(3) \
		--alpha $(5) \
		--tau $(6) \
		--factor $(7) \
		--adapt $(8) \
		--seed 42$(4) \
		--datafolder $(BUILD)/data/lastfm \
		--output $(BUILD)/results/$(1)/$(2)/$(3)/$(4).in_progress/
	mv $(BUILD)/results/$(1)/$(2)/$(3)/$(4).in_progress $(BUILD)/results/$(1)/$(2)/$(3)/$(4)
endef

# LastFM final tuned hyper parameters for each expeirments
$(BUILD)/results/lastfm/linear/avg/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,linear,avg,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/lastfm/linear/avg/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/linear/avg/$(i)/log.json)

$(BUILD)/results/lastfm/linear/exp/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,linear,exp,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/lastfm/linear/exp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/linear/exp/$(i)/log.json)

$(BUILD)/results/lastfm/linear/win/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,linear,win,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/linear/win/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/linear/win/$(i)/log.json)

$(BUILD)/results/lastfm/linear/adaexp/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,linear,adaexp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/linear/adaexp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/linear/adaexp/$(i)/log.json)

$(BUILD)/results/lastfm/linear/adawin/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,linear,adawin,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/linear/adawin/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/linear/adawin/$(i)/log.json)


$(BUILD)/results/lastfm/stationary/avg/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,stationary,avg,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/lastfm/stationary/avg/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/stationary/avg/$(i)/log.json)

$(BUILD)/results/lastfm/stationary/exp/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,stationary,exp,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/lastfm/stationary/exp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/stationary/exp/$(i)/log.json)

$(BUILD)/results/lastfm/stationary/win/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,stationary,win,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/stationary/win/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/stationary/win/$(i)/log.json)

$(BUILD)/results/lastfm/stationary/adaexp/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,stationary,adaexp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/stationary/adaexp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/stationary/adaexp/$(i)/log.json)

$(BUILD)/results/lastfm/stationary/adawin/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,stationary,adawin,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/stationary/adawin/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/stationary/adawin/$(i)/log.json)


$(BUILD)/results/lastfm/abrupt/avg/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,abrupt,avg,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/lastfm/abrupt/avg/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/abrupt/avg/$(i)/log.json)

$(BUILD)/results/lastfm/abrupt/exp/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,abrupt,exp,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/lastfm/abrupt/exp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/abrupt/exp/$(i)/log.json)

$(BUILD)/results/lastfm/abrupt/win/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,abrupt,win,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/abrupt/win/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/abrupt/win/$(i)/log.json)

$(BUILD)/results/lastfm/abrupt/adaexp/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,abrupt,adaexp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/abrupt/adaexp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/abrupt/adaexp/$(i)/log.json)

$(BUILD)/results/lastfm/abrupt/adawin/%/log.json : $(LASTFM_EST_DEPS) ; $(call perform_estimation,lastfm,abrupt,adawin,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/lastfm/abrupt/adawin/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/lastfm/abrupt/adawin/$(i)/log.json)


# Delicious final tuned hyper parameters for each expeirments
$(BUILD)/results/delicious/linear/avg/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,linear,avg,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/delicious/linear/avg/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/linear/avg/$(i)/log.json)

$(BUILD)/results/delicious/linear/exp/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,linear,exp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/delicious/linear/exp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/linear/exp/$(i)/log.json)

$(BUILD)/results/delicious/linear/win/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,linear,win,$(*),0.9999,50000,0.00005,500)
$(BUILD)/results/delicious/linear/win/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/linear/win/$(i)/log.json)

$(BUILD)/results/delicious/linear/adaexp/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,linear,adaexp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/delicious/linear/adaexp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/linear/adaexp/$(i)/log.json)

$(BUILD)/results/delicious/linear/adawin/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,linear,adawin,$(*),0.9999,50000,0.00001,300)
$(BUILD)/results/delicious/linear/adawin/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/linear/adawin/$(i)/log.json)


$(BUILD)/results/delicious/stationary/avg/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,stationary,avg,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/delicious/stationary/avg/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/stationary/avg/$(i)/log.json)

$(BUILD)/results/delicious/stationary/exp/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,stationary,exp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/delicious/stationary/exp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/stationary/exp/$(i)/log.json)

$(BUILD)/results/delicious/stationary/win/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,stationary,win,$(*),0.9999,50000,0.00005,500)
$(BUILD)/results/delicious/stationary/win/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/stationary/win/$(i)/log.json)

$(BUILD)/results/delicious/stationary/adaexp/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,stationary,adaexp,$(*),0.99995,10000,0.00005,1000)
$(BUILD)/results/delicious/stationary/adaexp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/stationary/adaexp/$(i)/log.json)

$(BUILD)/results/delicious/stationary/adawin/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,stationary,adawin,$(*),0.9999,50000,0.00001,300)
$(BUILD)/results/delicious/stationary/adawin/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/stationary/adawin/$(i)/log.json)


$(BUILD)/results/delicious/abrupt/avg/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,abrupt,avg,$(*),0.9999,10000,0.00005,500)
$(BUILD)/results/delicious/abrupt/avg/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/abrupt/avg/$(i)/log.json)

$(BUILD)/results/delicious/abrupt/exp/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,abrupt,exp,$(*),0.99995,10000,0.00005,500)
$(BUILD)/results/delicious/abrupt/exp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/abrupt/exp/$(i)/log.json)

$(BUILD)/results/delicious/abrupt/win/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,abrupt,win,$(*),0.9999,50000,0.00005,500)
$(BUILD)/results/delicious/abrupt/win/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/abrupt/win/$(i)/log.json)

$(BUILD)/results/delicious/abrupt/adaexp/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,abrupt,adaexp,$(*),0.99995,10000,0.00005,1000)
$(BUILD)/results/delicious/abrupt/adaexp/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/abrupt/adaexp/$(i)/log.json)

$(BUILD)/results/delicious/abrupt/adawin/%/log.json : $(DELICIOUS_EST_DEPS) ; $(call perform_estimation,delicious,abrupt,adawin,$(*),0.9999,50000,0.00001,300)
$(BUILD)/results/delicious/abrupt/adawin/ : $(foreach i,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20,$(BUILD)/results/delicious/abrupt/adawin/$(i)/log.json)


# Recipes to make everything
.PHONY: lastfm/results delicious/results
lastfm/results : $(BUILD)/results/lastfm/linear/avg/
lastfm/results : $(BUILD)/results/lastfm/linear/exp/
lastfm/results : $(BUILD)/results/lastfm/linear/win/
lastfm/results : $(BUILD)/results/lastfm/linear/adaexp/
lastfm/results : $(BUILD)/results/lastfm/linear/adawin/
lastfm/results : $(BUILD)/results/lastfm/stationary/avg/
lastfm/results : $(BUILD)/results/lastfm/stationary/exp/
lastfm/results : $(BUILD)/results/lastfm/stationary/win/
lastfm/results : $(BUILD)/results/lastfm/stationary/adaexp/
lastfm/results : $(BUILD)/results/lastfm/stationary/adawin/
lastfm/results : $(BUILD)/results/lastfm/abrupt/avg/
lastfm/results : $(BUILD)/results/lastfm/abrupt/exp/
lastfm/results : $(BUILD)/results/lastfm/abrupt/win/
lastfm/results : $(BUILD)/results/lastfm/abrupt/adaexp/
lastfm/results : $(BUILD)/results/lastfm/abrupt/adawin/
delicious/results : $(BUILD)/results/delicious/linear/avg/
delicious/results : $(BUILD)/results/delicious/linear/exp/
delicious/results : $(BUILD)/results/delicious/linear/win/
delicious/results : $(BUILD)/results/delicious/linear/adaexp/
delicious/results : $(BUILD)/results/delicious/linear/adawin/
delicious/results : $(BUILD)/results/delicious/stationary/avg/
delicious/results : $(BUILD)/results/delicious/stationary/exp/
delicious/results : $(BUILD)/results/delicious/stationary/win/
delicious/results : $(BUILD)/results/delicious/stationary/adaexp/
delicious/results : $(BUILD)/results/delicious/stationary/adawin/
delicious/results : $(BUILD)/results/delicious/abrupt/avg/
delicious/results : $(BUILD)/results/delicious/abrupt/exp/
delicious/results : $(BUILD)/results/delicious/abrupt/win/
delicious/results : $(BUILD)/results/delicious/abrupt/adaexp/
delicious/results : $(BUILD)/results/delicious/abrupt/adawin/
