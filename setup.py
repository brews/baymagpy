from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='baymagpy',
    version='0.0.1a1',
    description='Calibration of Mg/Ca records using Bayesian regression',
    long_description=readme(),
    long_description_content_type="text/markdown",
    license='GPLv3',
    author='S. Brewster Malevich',
    author_email='malevich@email.arizona.edu',
    url='https://github.com/brews/baymagpy',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='marine paleoclimate calibration mgca',

    packages=find_packages(exclude=['docs']),


    install_requires=['numpy', 'co2syspy', 'gsw', 'attrs', 'xarray', 'pandas',
                      'matplotlib', 'scipy', 'netcdf4', 'shapely'],
    tests_require=['pytest'],
    package_data={'baymag': ['modelparams/tracedumps/*.csv',
                             'omega/observations/*.nc',
                             'omega/observations/*.mat']},
)
