install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	python3 -m black .

lint:
	python3 -m pylint --generated-member=torch.cat --disable=R,C ddpg/**/*.py

test:
	python3 -m pytest -vv --cov

all: install lint test