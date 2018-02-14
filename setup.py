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
    python_requires='>=3.5',
    install_requires = [],
    extras_require = {
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    cmdclass = {'test': RunTests},
)
