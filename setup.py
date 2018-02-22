"""Packaging settings."""


from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from qlknn import __version__


this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test', '--cov=qlknn', '--cov-report=term-missing',
                      '--ignore=lib/'])
        raise SystemExit(errno)

nndb_require = ['peewee>=3.0.16', 'psycopg2']
training_require = ['tensorflow>=1.3']
plot_require = nndb_require + training_require
pipeline_require = ['luigi']
all_require = nndb_require + training_require + plot_require + pipeline_require

setup(
    name = 'qlknn',
    version = __version__,
    description = 'Tools to create QuaLiKiz Quasi-linear gyrokinetic code Neural Networks',
    long_description = long_description,
    url = 'https://github.com/Karel-van-de-Plassche/QLKNN-develop',
    author = 'Karel van de Plassche',
    author_email = 'karelvandeplassche@gmail.com',
    license = 'MIT',
    classifiers = [
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: MIT',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords = '',
    packages = find_packages(exclude=['docs', 'tests*']),
    python_requires='>=3.4',
    install_requires = ['ipython', 'numpy', 'scipy', 'xarray', 'pandas>=0.15.2'],
    extras_require = {
        'test': ['coverage', 'pytest', 'pytest-cov'],
        'nndb': nndb_require,
        'training': training_require,
        'pipeline': pipeline_require,
        'plot': plot_require,
        'all': all_require
    },
    cmdclass = {'test': RunTests},
)
