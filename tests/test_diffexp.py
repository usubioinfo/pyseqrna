import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from pyseqrna.modules.diffexp.deseq2 import DESeq2DiffExp

@pytest.fixture
def mock_deseq2_config():
    return {
        'deseq2': {
            'fitType': 'parametric'
        }
    }

# DESeq2DiffExp does not have load_config.
# It calls _load_data in BaseDiffExp.
# We should mock _load_data or provide dummy files that pass validation.
# Or mock FileManager.verify_files_exist.

@patch('pyseqrna.modules.diffexp.deseq2.DESeq2DiffExp._check_r_dependencies')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_deseq2_initialization(mock_verify, mock_check_r, mock_deseq2_config, temp_outdir):
    """Test DESeq2 initialization."""
    mock_verify.return_value = True

    # We need to mock _load_file_as_dataframe or provide actual files?
    # BaseDiffExp._load_data calls _load_file_as_dataframe.

    with patch('pyseqrna.modules.diffexp.base.BaseDiffExp._load_file_as_dataframe') as mock_load_df:
        mock_load_df.return_value = pd.DataFrame({'Gene': ['G1'], 'S1': [10], 'condition': ['A']})

        de = DESeq2DiffExp(count_matrix_file='dummy.csv', sample_info_file='dummy.csv', comparisons=['A-B'], out_dir=temp_outdir)

        assert de.tool_name == 'deseq2'
        mock_check_r.assert_called_once()

@patch('pyseqrna.modules.diffexp.deseq2.DESeq2DiffExp._check_r_dependencies')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_run_analysis_mock(mock_verify, mock_check_r, mock_deseq2_config, temp_outdir, sample_counts_df):
    """Test DESeq2 run method with mocked R execution."""
    mock_verify.return_value = True

    with patch('pyseqrna.modules.diffexp.base.BaseDiffExp._load_file_as_dataframe') as mock_load_df:
        # Mock count data and sample data
        mock_load_df.side_effect = [
            sample_counts_df, # count matrix (init)
            pd.DataFrame({'sample': ['CondA_1', 'CondA_2', 'CondB_1', 'CondB_2'], 'condition': ['CondA', 'CondA', 'CondB', 'CondB']}), # sample info (init)
            sample_counts_df, # count matrix (run)
            pd.DataFrame({'sample': ['CondA_1', 'CondA_2', 'CondB_1', 'CondB_2'], 'condition': ['CondA', 'CondA', 'CondB', 'CondB']}) # sample info (run)
        ]

        de = DESeq2DiffExp(count_matrix_file='dummy.csv', sample_info_file='dummy.csv', comparisons=['CondA-CondB'], out_dir=temp_outdir)

        # Mock internal DESeq2 execution with the full result schema expected by the wrapper
        de._run_deseq2_analysis = MagicMock(return_value=pd.DataFrame({
            'Gene': ['GeneA'],
            'baseMean': [55.0],
            'logFC': [1.0],
            'lfcSE': [0.2],
            'stat': [5.0],
            'pvalue': [1e-4],
            'FDR': [0.05],
        }))

        # Run analysis
        results = de.run(save_results=False, filter_results=False)

        assert 'combined_results' in results['results']
