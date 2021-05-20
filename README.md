<p align="center">
<a href="https://badge.fury.io/py/oogeso"><img src="https://badge.fury.io/py/oogeso.svg"></a>
<a href="https://github.com/oogeso/oogeso/actions?query=workflow%3ACI"><img src="https://img.shields.io/github/workflow/status/oogeso/oogeso/CI"></a>
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

Clone or download the code and install it as a python package. I.e. navigate to the folder with the MANIFEST.in file and type:  
`pip install --editable .`

(Installing as _editable_ means a link will be created from the python package directory to the code located here, and you can modify the code without the
need to re-install.)

Check out the examples:

* [Simple test case](examples/test case2.ipynb?viewer=nbviewer)
* [Test oil and gas platform](examples/TestPlatform.ipynb?viewer=nbviewer)

## User guide
The online user guide  gives more information about how to
specify input data and run a simulation case.

*  [User guide](userguide.md)

There is also a (not always up-to-date) manual with more information and explanations
about the modelling concepts and model elements:

* [Manual (pdf)](../../raw/master/doc/oogeso_manual.pdf)

## Contribute
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches to avoid messing things up -- but don't veer too far away from the trunk (master branch)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research
