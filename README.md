<p>
<a href="https://badge.fury.io/gh/oogeso%2Foogeso"><img src="https://badge.fury.io/gh/oogeso%2Foogeso.svg" alt="GitHub version" height="18"></a>
<a href="https://github.com/oogeso/oogeso/actions/workflows/build.yml?query=workflow%3ACI"><img src="https://img.shields.io/github/workflow/status/oogeso/oogeso/CI" alt="Badge"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg" alt="Badge"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Badge"></a>
<a href="https://lgtm.com/projects/g/oogeso/oogeso/alerts/"><img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/oogeso/oogeso.svg?logo=lgtm&logoWidth=18"/></a>
<a href="https://lgtm.com/projects/g/oogeso/oogeso/context:python"><img src="https://img.shields.io/lgtm/grade/python/g/oogeso/oogeso.svg?logo=lgtm&logoWidth=18" alt="Badge"></a>
</p>
<br/>

# Offshore Oil and Gas Energy System Operational Optimisation Model (oogeso)

Python module for modelling and analysing the energy system of offshore oil and gas fields, with renewable energy and storage integration.

Part of the [Low Emission Centre](https://www.sintef.no/en/projects/lowemission-research-centre/) (SP5).

## Getting started
Install latest Oogeso release from PyPi:
```
pip install oogeso
```

in order to use the plotting functionality you will need to install plotting libraries:

```
pip install matplotlib plotly seaborn
```

## User guide and examples
The online user guide  gives more information about how to
specify input data and run a simulation case.

*  [User guide](https://oogeso.github.io/oogeso/)


## Local installation
Prerequisite: 
- [Poetry](https://python-poetry.org/docs/#installation)
- [Pre-commit](https://pre-commit.com/)
- [CBC solver](https://projects.coin-or.org/Cbc)
Clone or download the code and install it as a python package. I.e. navigate to the folder with the MANIFEST.in file and type:

### Install dependencies
1. `git clone git@github.com:oogeso/oogeso.git`
2. `cd oogeso`
3. `poetry install --no-root`  --no-root to not install the package itself, only the dependencies.
4. `poetry shell`
5. `poetry run pytest tests`

### Local development in Docker
Alternatively you can run and develop the code using docker and the Dockerfile in the root folder.

### GitHub Actions Pipelines
4 pipelines are defined.

1. Build: Building and testing on multiple OS and python versions. Triggered on any push to GitHub.
2. CBC-optimizer CI: Build and test oogeso with the CBC-solver and spesific cbc-tests.
3. Release: Create release based on tags starting on v*.
4. Publish: Publish the package to PyPi when a release is marked as published.

## Contribute
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches for development and pull requests to merge into main
* Use [Pre-commit hooks](https://pre-commit.com/)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research
