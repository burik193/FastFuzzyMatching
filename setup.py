from setuptools import setup, find_packages
import json
import os


def read_pipenv_dependencies(fname):
    """Get Pipfile.lock dependencies."""
    filepath = os.path.join(os.path.dirname(__file__), fname)
    with open(filepath) as lockfile:
        lockjson = json.load(lockfile)
        return [dependency for dependency in lockjson.get('default')]

setup(
    name='FastFuzzyMatching',
    version=os.getenv('PACKAGE VERSION', '1.0.0a'),
    package_dir={'': 'fuzzyMatcher'},
    packages=find_packages('fuzzyMatcher', include=['matcher*']),
    url='',
    license='',
    author='Oleg Burik',
    author_email='M302242@one.merckgroup.com',
    description='Fast simple matching of two datasets of different length',
    install_requires=[*read_pipenv_dependencies('Pipfile.lock')]
)
