FROM python:3.9.8 as dev

ARG INSTALL_DEV=true
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
  apt-get -y upgrade && \
  rm -rf /var/lib/apt/lists/* \

WORKDIR /coinbrew

# Download repository with CBC solver and intstall
RUN git clone https://github.com/coin-or/coinbrew /var/cbc
WORKDIR /var/cbc
RUN ./coinbrew fetch Cbc:stable/2.10 --no-prompt --no-third-party
RUN ./coinbrew build Cbc --no-prompt --no-third-party --prefix=/usr
ENV COIN_INSTALL_DIR /usr

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN pip install poetry && poetry config virtualenvs.create false

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN bash -c "if [ INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

ENV PYTHONPATH=/code

FROM dev AS build

COPY ./ ./

RUN poetry install --no-dev
