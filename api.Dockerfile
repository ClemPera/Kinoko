FROM python:3.12.9

# COPY . /app
COPY api /app/api
COPY models /app/models
COPY requirements_api.txt /app

WORKDIR /app
RUN pip install -r requirements_api.txt

CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT