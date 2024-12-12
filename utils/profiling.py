from argparse import Action
from ast import main
import cProfile
from distutils.core import setup
from json.tool import main
import pstats
from turtle import up

from pkg_resources import Requirement
from agents.code_transformer import CodeTransformer

def profile_transform():
    transformer = CodeTransformer()
    with cProfile.Profile() as pr:
        transformer.transform_code("def foo(): pass")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()

if __name__ == "__main__":
    profile_transform() name: CI Pipeline

on: 
  push:main
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:upsetup
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r Requirement.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest --cov=./name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: Action/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest --cov=./