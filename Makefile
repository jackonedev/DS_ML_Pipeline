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
