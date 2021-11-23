<p>
<a href="https://badge.fury.io/gh/oogeso%2Foogeso"><img src="https://badge.fury.io/gh/oogeso%2Foogeso.svg" alt="GitHub version" height="18"></a>
<a href="https://github.com/oogeso/oogeso/actions/workflows/build.yml?query=workflow%3ACI"><img src="https://img.shields.io/github/workflow/status/oogeso/oogeso/CI"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9-blue.svg"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://lgtm.com/projects/g/oogeso/oogeso/alerts/"><img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/oogeso/oogeso.svg?logo=lgtm&logoWidth=18"/></a>
<a href="https://lgtm.com/projects/g/oogeso/oogeso/context:python"><img src="https://img.shields.io/lgtm/grade/python/g/oogeso/oogeso.svg?logo=lgtm&logoWidth=18"></a>
<a href="https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fharald_g_svendsen%2Foogeso/HEAD?filepath=examples"><img src="https://mybinder.org/badge_logo.svg"></a>
</p>
<br/>

# Offshore Oil and Gas Energy System Operational Optimisation Model (oogeso)

Python module for modelling and analysing the energy system of offshore oil and gas fields, with renewable energy and storage integration.

Part of the [Low Emission Centre](https://www.sintef.no/en/projects/lowemission-research-centre/) (SP5).

## Getting started

Pypi distribution to come. See local installation below.

## Local installation
Prerequisite: [Poetry](https://python-poetry.org/docs/#installation)

Clone or download the code and install it as a python package. I.e. navigate to the folder with the MANIFEST.in file and type:

### Install as a package for normal use:
1. `poetry install`

### Install dependencies for local development
2. `poetry install --no-root`  --no-root to not install the package itself, only the dependencies.
3. `poetry shell`
4. `poetry run pytests tests`

### Local development in Docker
Alternatively you can run and develop the code using docker and the Dockerfile in the root folder.

## User guide
The online user guide  gives more information about how to
specify input data and run a simulation case.

*  [User guide](userguide.md)

There is also a (not always up-to-date) manual with more information and explanations
about the modelling concepts and model elements:

* [Manual (pdf)](../../raw/master/doc/oogeso_manual.pdf)

## Examples
Check out the examples:

* [Simple test case](examples/test case2.ipynb?viewer=nbviewer)
* [Test oil and gas platform](examples/TestPlatform.ipynb?viewer=nbviewer)

## GitHub Actions Pipelines
3 pipelines are defined.

1. Build: Building and testing on multiple OS and python versions. Triggered on any push to GitHub.
2. Release: Create release based on tags starting on v*.
3. Publish: Publish package to PyPi.

## Contribute
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches to avoid messing things up -- but don't veer too far away from the trunk (master branch)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research
