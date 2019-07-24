# Copyright (c) 2019 Doug Napoleone, All rights reserved.
from setuptools import setup, find_packages
import version

setup(
    name="trtis_cidmgr",
    version=version.__version__,
    description="FogBugz API Object Relational Mapper (ORM)",
    long_description=open('README.md').read(),
    url="https://github.com/dougn/trtis-cidmgr/",
    author=version.__author__,
    author_email=version.__email__,
    license="BSD",
    packages=["trtis_cidmgr"],
    entry_points = {
        'console_scripts': [
            'trtis-cidmgr-model=trtis_cidmgr.model:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        'numpy',
        'protobuf==3.6.0', # limit to 3.6 or things break.
        'grpcio==1.19.0', # limit to 1.19 due to trtserver issues
        'grpcio-tools==1.19.0', # limit to 1.19 due to trtserver issues
        'tensorrtserver>=1.5.0.dev0'],
    keywords="tensorrt inference server service client".split(),
    zip_safe=True
)
