FROM bitnami/python:3.10 as dev

ARG INSTALL_DEV=true
ENV PYTHONUNBUFFERED=1

RUN apt update && apt install -y unzip git gcc g++ gfortran make wget file pkg-config libblas-dev liblapack-dev
WORKDIR /coinbrew

# Download repository with CBC solver and intstall
RUN git clone https://github.com/coin-or/coinbrew /var/cbc
WORKDIR /var/cbc
RUN ./coinbrew fetch Cbc:stable/2.10 --no-prompt --no-third-party && \
  ./coinbrew build Cbc --no-prompt --no-third-party --prefix=/usr
ENV COIN_INSTALL_DIR /usr

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN pip install poetry && poetry install matplotlib plotly seaborn && poetry config virtualenvs.create false

RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

FROM dev as test

COPY ./ ./
ENV PYTHONPATH=/code/src

FROM test AS build

RUN poetry install --no-dev


