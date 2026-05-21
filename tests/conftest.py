import pytest
import pandas as pd
import numpy as np
import os
import shutil

@pytest.fixture
def sample_deg_df():
    """
    Creates a sample DataFrame mimicking differential expression results.
    """
    data = {
        'Gene': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
        'logFC(CondA-CondB)': [2.5, -2.5, 0.1, 1.5, -1.5],
        'pvalue(CondA-CondB)': [0.001, 0.001, 0.5, 0.04, 0.04],
        'FDR(CondA-CondB)': [0.005, 0.005, 0.8, 0.045, 0.045]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_counts_df():
    """
    Creates a sample DataFrame mimicking normalized counts.
    """
    data = {
        'Gene': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
        'CondA_1': [100, 10, 50, 80, 20],
        'CondA_2': [110, 12, 55, 85, 22],
        'CondB_1': [10, 100, 50, 20, 80],
        'CondB_2': [12, 110, 55, 22, 85]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_outdir(tmp_path):
    """
    Fixture for a temporary output directory.
    """
    d = tmp_path / "output"
    d.mkdir()
    return str(d)
