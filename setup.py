#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for PySeqRNA
========================

A Python-based RNA-seq data analysis package with comprehensive
quality control, trimming, alignment, quantification, and analysis tools.

Author: Naveen Duhan
"""

from pathlib import Path

from setuptools import setup, find_packages

version_ns = {}
exec((Path(__file__).parent / "pyseqrna" / "__version__.py").read_text(), version_ns)

setup(
    name="pyseqrna",
    version=version_ns["__version__"],
    author="Naveen Duhan",
    author_email="naveen.duhan@usu.edu",
    description="A Python-based RNA-seq data analysis package",
    long_description="PySeqRNA: A comprehensive RNA-seq data analysis package with quality control, trimming, alignment, quantification, and analysis tools.",
    url="https://github.com/kaundal-lab/pyseqrna",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "psutil>=5.8.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "adjustText>=0.7.3",
        "requests>=2.25.0",
        "anndata>=0.8.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "openpyxl>=3.0.0",
        "tabulate>=0.9.0",
        "python-docx>=1.1.0",
        "future>=0.18.3",
        "patsy>=0.5.0",
        "urllib3>=1.26.0",
        "importlib-resources>=5.0.0",
        "pysam>=0.22.0",
        "pyfastx>=2.1.0",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "pyseqrna=pyseqrna.__main__:main",
        ],
    },
    package_data={
        "pyseqrna": ["param/*.ini"],
    },
    include_package_data=True,
    zip_safe=False,
)
