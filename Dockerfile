FROM python:3.9 as dev

RUN apt-get update && \
  apt-get -y upgrade && \
  rm -rf /var/lib/apt/lists/* \

ARG INSTALL_DEV=false
ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN pip install poetry && poetry config virtualenvs.create false

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN bash -c "if [ INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

ENV PYTHONPATH=/code

FROM build AS tests

COPY ./ ./

RUN poetry install --no-dev
