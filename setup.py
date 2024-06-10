#!/usr/bin/env python3
"""Modelitool"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="modelitool",
    version="0.2",
    description="Tools for Modelica",
    long_description=long_description,
    url="https://github.com/BuildingEnergySimulationTools/modelitool",
    author="Nobatek/INEF4",
    author_email="bdurandestebe@nobatek.com",
    license="License :: OSI Approved :: BSD License",
    # keywords=[
    # ],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.17.3",
        "OMPython>=3.5.2",
        "corrai>=0.3.0",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
