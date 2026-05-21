#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySeqRNA Modules Package
========================

This package contains all the core modules for PySeqRNA:

- alignment: Read alignment modules (STAR, Bowtie2, BWA, etc.)
- annotation: Gene annotation and feature extraction modules
- clustering: Sample similarity clustering modules
- coexpression: Gene co-expression analysis modules
- diffexp: Differential expression analysis modules (negative binomial, empirical Bayes, etc.)
- multimapped_groups: Multimapped read groups analysis modules
- normalization: Count normalization modules (median ratio, TMM, etc.)
- quality: Quality control and assessment modules
- quantification: Feature counting and quantification modules
- reporting: Comprehensive run report generation modules
- trimming: Read trimming and preprocessing modules
- visualization: Data visualization and plotting modules

:Created: May 20, 2021
:Updated: May 20, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from . import quality
from . import trimming
from . import normalization
from . import diffexp
from . import multimapped_groups
from . import clustering
from . import coexpression
from . import reporting
from pyseqrna.__version__ import __version__

__author__ = "Naveen Duhan"

__all__ = [
    "quality",
    "trimming",
    "alignment",
    "normalization",
    "diffexp",
    "multimapped_groups",
    "clustering",
    "coexpression",
    "reporting",
]
