#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lightning_quick',
    version='0.0.1',
    description='pytorch-lightning based template to quickly implement new ideas',
    author='Theo J. Adrai',
    author_email='tjtadrai@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/theoad/quickstart',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Amazon Copyright",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pytorch-lightning>=1.6.4',
        'torchmetrics>=0.9.2',
        'wandb>=0.12.21'
    ],
    python_requires='>=3.8',
)
