FROM python:3.9 as dev
RUN apt-get update && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip poetry && poetry config virtualenvs.create false && poetry install --no-root

FROM dev AS build

COPY ./ ./
RUN pip install --upgrade pip poetry && poetry config virtualenvs.create false && poetry install --optional

FROM build AS tests

RUN poetry run pytest tests
