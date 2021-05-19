from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

REQUIREMENTS = [
    "dash",
	"flask",
	"matplotlib",
	"networkx",
    "numpy",
    "pandas",
	"plotly",
	"pydot",
	"pyyaml",
	"Pyomo",
    "scipy",
	"seaborn",
	"tqdm",
	"xlrd",
]

TEST_REQUIRES = [
    "black",
    "mypy",
    "pylint",
    "pytest",
]

setup(name='oogeso',
      install_requires=REQUIREMENTS,
	  tests_require=TEST_REQUIRES,
	  setup_requires=["setuptools_scm~=3.2"],      
      description='Offshore Oil and Gas Field Energy System Operational Optimisation (OOGESO)',
	  long_description=LONG_DESCRIPTION,
	  url="https://github.com/oogeso/oogeso",
	  use_scm_version=True,
      license='MIT License (http://opensource.org/licenses/MIT)',
      package_dir={"": "src"},
	  packages=find_packages("src"),
      zip_safe = False,
	  classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
		'License :: OSI Approved :: MIT License',
	  ],
	  keywords = 'offshore energy system, oil and gas, operational optimisation',
	 )
