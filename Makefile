PYTHON=python
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000

.PHONY: train serve docker-build docker-up docker-down test lint

train:
	$(PYTHON) train.py

serve:
	uvicorn api.main:app --host $(UVICORN_HOST) --port $(UVICORN_PORT) --reload

docker-build:
	docker build -t stock-lstm-api:latest .

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile model/*.py api/*.py train.py
	echo "Syntax OK"
