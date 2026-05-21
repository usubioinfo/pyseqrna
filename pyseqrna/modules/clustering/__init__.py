#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering module for PySeqRNA.

This package contains sample similarity clustering. Gene co-expression
analysis lives in ``pyseqrna.modules.coexpression``.

:Created: May 20, 2021
:Updated: April 5, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .base import ClusteringError
from .native import ClusteringAnalyzer

__all__ = ["ClusteringAnalyzer", "ClusteringError"]
