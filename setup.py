#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
]

test_requirements = [
    "pytest",
]


setup(
    name='rep',
    version='0.0.1',
    description="rep: RNA Expression prediction",
    author="Florian Hölzlwimmer",
    author_email='git.ich@frhoelzlwimmer.de',
    url='https://gitlab.cmm.in.tum.de/gagneurlab/REP',
    long_description="rep: toolkit for seq2seq models in genomics",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": [
            "bumpversion",
            "wheel",
            "jedi",
            "epc",
            "pytest",
            "pytest-pep8",
            "pytest-cov"
        ],
        "dask": [
            "dask",
            "joblib",
        ],
        "ray": [
            "ray",
        ],
        "spark": [
            "pyspark",
            "pyarrow",
        ],
        "polars": [
            "polars~=0.20.4",
            "pyarrow",
        ]
    },
    entry_points={'console_scripts': ['rep = rep.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    package_data={'rep': ['logging.conf']},
    tests_require=test_requirements
)
