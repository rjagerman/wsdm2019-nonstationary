# Recipes for training policies
.PRECIOUS: $(BUILD)/models/%.gz

LASTFM_TRAIN_DEPS := $(BUILD)/data/lastfm/bandit_items.gz
LASTFM_TRAIN_DEPS += $(BUILD)/data/lastfm/bandit_users.gz
LASTFM_TRAIN_DEPS += $(BUILD)/data/lastfm/bandit_clusters.gz

DELICIOUS_TRAIN_DEPS := $(BUILD)/data/delicious/bandit_items.gz
DELICIOUS_TRAIN_DEPS += $(BUILD)/data/delicious/bandit_users.gz
DELICIOUS_TRAIN_DEPS += $(BUILD)/data/delicious/bandit_clusters.gz

$(BUILD)/models/lastfm/logger.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_logger,lastfm)
$(BUILD)/models/lastfm/candidate_0.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,0,lastfm)
$(BUILD)/models/lastfm/candidate_1.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,1,lastfm)
$(BUILD)/models/lastfm/candidate_2.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,2,lastfm)
$(BUILD)/models/lastfm/candidate_3.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,3,lastfm)
$(BUILD)/models/lastfm/candidate_4.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,4,lastfm)
$(BUILD)/models/lastfm/candidate_5.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,5,lastfm)
$(BUILD)/models/lastfm/candidate_6.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,6,lastfm)
$(BUILD)/models/lastfm/candidate_7.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,7,lastfm)
$(BUILD)/models/lastfm/candidate_8.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,8,lastfm)
$(BUILD)/models/lastfm/candidate_9.gz : $(LASTFM_TRAIN_DEPS) ; $(call train_cluster,9,lastfm)

$(BUILD)/models/delicious/logger.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_logger,delicious)
$(BUILD)/models/delicious/candidate_0.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,0,delicious)
$(BUILD)/models/delicious/candidate_1.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,1,delicious)
$(BUILD)/models/delicious/candidate_2.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,2,delicious)
$(BUILD)/models/delicious/candidate_3.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,3,delicious)
$(BUILD)/models/delicious/candidate_4.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,4,delicious)
$(BUILD)/models/delicious/candidate_5.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,5,delicious)
$(BUILD)/models/delicious/candidate_6.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,6,delicious)
$(BUILD)/models/delicious/candidate_7.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,7,delicious)
$(BUILD)/models/delicious/candidate_8.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,8,delicious)
$(BUILD)/models/delicious/candidate_9.gz : $(DELICIOUS_TRAIN_DEPS) ; $(call train_cluster,9,delicious)

define train_logger
	mkdir -p $(dir $@)
	$(PYTHON) -m experiments.hetrec.train \
		--seed 42 \
		--datafolder $(BUILD)/data/$(1)/ \
		--iterations 100000 \
		--save $@.in_progress
	mv $@.in_progress $@
endef

define train_cluster
	mkdir -p $(dir $@)
	$(PYTHON) -m experiments.hetrec.train \
		--seed 42 \
		--datafolder $(BUILD)/data/$(2)/ \
		--iterations 100000 \
		--specialized $(1) \
		--save $@.in_progress
	mv $@.in_progress $@
endef

.PHONY: lastfm/candidates delicious/candidates
lastfm/logger : $(BUILD)/models/lastfm/logger.gz
lastfm/candidates : $(foreach index,0 1 2 3 4 5 6 7 8 9,$(BUILD)/models/lastfm/candidate_$(index).gz)
delicious/logger : $(BUILD)/models/delicious/logger.gz
delicious/candidates : $(foreach index,0 1 2 3 4 5 6 7 8 9,$(BUILD)/models/delicious/candidate_$(index).gz)
