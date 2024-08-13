install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# test:
# 	python -m pytest -vv tests/*.py

format:
	isort --profile=black . &&\
	autopep8 --in-place ./*.py tools/*.py utils/*.py ds_pipe/*.py ml_pipe/*.py &&\
	black --line-length 88 .

lint:
	pylint --disable=R,C *.py tools/*.py utils/*.py ds_pipe/*.py ml_pipe/*.py

DS_PIPE_FOLDER := ds_pipe
FEATURES_FOLDER := features
REPORTS_FOLDER := reports

TODAY := $(shell date +%Y-%m-%d)
DATED_FOLDER := $(TODAY)_attempt_
STORAGE_DIR := $(shell pwd)/ds_versions
NEXT_NUM := $(shell bash next_attempt.sh $(STORAGE_DIR) $(DATED_FOLDER))
DATED_ENUM_FOLDER := $(DATED_FOLDER)$(NEXT_NUM)
DATED_ENUM_DIR := $(STORAGE_DIR)/$(DATED_ENUM_FOLDER)

create_attempt_folder:
	mkdir -p $(DATED_ENUM_DIR)

copy_ds_pipe: create_attempt_folder
	rsync -av --exclude='__pycache__' $(DS_PIPE_FOLDER)/ $(DATED_ENUM_DIR)/$(DS_PIPE_FOLDER)/

copy_features_folder: create_attempt_folder
	cp -r $(FEATURES_FOLDER) $(DATED_ENUM_DIR)/$(FEATURES_FOLDER)
	
copy_reports_folder: create_attempt_folder
	cp -r $(REPORTS_FOLDER) $(DATED_ENUM_DIR)/$(REPORTS_FOLDER)

new_attempt: create_attempt_folder copy_ds_pipe copy_features_folder copy_reports_folder
