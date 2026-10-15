#!/usr/bin/env python
from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="logitfill",
    version="0.1.0",
    description="LogitFill: Key-channel mixed precision by attention-logit water-filling",
    author="LogitFill",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
