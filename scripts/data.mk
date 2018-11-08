# Recipes for data downloading and processing
.PRECIOUS: $(BUILD)/data/%.zip
.PRECIOUS: $(BUILD)/data/%.dat
.PRECIOUS: $(BUILD)/data/%.gz

$(BUILD)/data/delicious.zip :
	mkdir -p $(dir $@)
	wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip -O $@

$(BUILD)/data/lastfm.zip :
	mkdir -p $(dir $@)
	wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip -O $@

$(BUILD)/data/delicious/%.dat : $(BUILD)/data/delicious/tags.dat ;
$(BUILD)/data/delicious/tags.dat : $(BUILD)/data/delicious.zip
	mkdir -p $(dir $@)
	unzip -DDo $(BUILD)/data/delicious.zip -d $(BUILD)/data/delicious/

$(BUILD)/data/lastfm/%.dat : $(BUILD)/data/lastfm/tags.dat ;
$(BUILD)/data/lastfm/tags.dat : $(BUILD)/data/lastfm.zip
	mkdir -p $(dir $@)
	unzip -DDo $(BUILD)/data/lastfm.zip -d $(BUILD)/data/lastfm/

LASTFM_DATA_DEPS := $(BUILD)/data/lastfm/tags.dat
LASTFM_DATA_DEPS += $(BUILD)/data/lastfm/user_taggedartists.dat
LASTFM_DATA_DEPS += $(BUILD)/data/lastfm/user_artists.dat
LASTFM_DATA_DEPS += $(BUILD)/data/lastfm/user_friends.dat

$(BUILD)/data/lastfm/bandit_items.gz : $(BUILD)/data/lastfm/bandit_clusters.gz ;
$(BUILD)/data/lastfm/bandit_users.gz : $(BUILD)/data/lastfm/bandit_clusters.gz ;
$(BUILD)/data/lastfm/bandit_clusters.gz : $(LASTFM_DATA_DEPS) ; $(call preprocess_data,lastfm)

DELICIOUS_DATA_DEPS := $(BUILD)/data/delicious/tags.dat
DELICIOUS_DATA_DEPS += $(BUILD)/data/delicious/bookmark_tags.dat
DELICIOUS_DATA_DEPS += $(BUILD)/data/delicious/user_taggedbookmarks.dat
DELICIOUS_DATA_DEPS += $(BUILD)/data/delicious/user_contacts.dat

$(BUILD)/data/delicious/bandit_items.gz : $(BUILD)/data/delicious/bandit_clusters.gz ;
$(BUILD)/data/delicious/bandit_users.gz : $(BUILD)/data/delicious/bandit_clusters.gz ;
$(BUILD)/data/delicious/bandit_clusters.gz : $(DELICIOUS_DATA_DEPS) ; $(call preprocess_data,delicious)

define preprocess_data
	$(PYTHON) -m experiments.preprocess.$(1) \
		--seed 42 \
		--datafolder $(BUILD)/data/$(1)/ \
		--clusters 10 \
		--top_clusters 10
endef
