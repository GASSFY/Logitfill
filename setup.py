#!/usr/bin/env python
from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="logitfill",
    version="0.2.0",
    description="LogitFill: offline Key policy and uniform Value quantization",
    author="LogitFill",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    package_data={"logitfill": ["packed/*.cu"]},
    include_package_data=True,
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
