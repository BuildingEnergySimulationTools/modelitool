#!/usr/bin/env python3
"""Modelitool"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="modelitool",
    version="0.1",
    description="Tools for Modelica",
    long_description=long_description,
    # url="",
    author="Nobatek/INEF4",
    author_email="bdurandestebe@nobatek.com",
    # license="",
    # keywords=[
    # ],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.2.3",
        "numpy>=1.17.3",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)