from setuptools import setup

exec(open('ogeso/version.py').read())

setup(name='ogeso',
      version=__version__,
      description='Offshore Oil and Gas Field Energy System Operational Optimisation (OGESO)',
      license='MIT License (http://opensource.org/licenses/MIT)',
      packages=['ogeso'],
      zip_safe = True,
	  classifiers = [
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3'
	  ],
	  keywords = 'offshore energy system, oil and gas, operational optimisation',
	 )
