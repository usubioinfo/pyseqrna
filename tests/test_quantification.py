import pytest
import os
from unittest.mock import patch, MagicMock
from pyseqrna.modules.quantification.featurecounts import FeatureCountsQuantifier as FeatureCounts

@pytest.fixture
def mock_fc_config():
    return {
        'featureCount': { # Note: Config section name might be featureCount or featurecounts
            '-T': '4',
            '-p': ''
        }
    }

@patch('shutil.which')
@patch('pyseqrna.modules.quantification.featurecounts.FeatureCountsQuantifier.load_config')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_featurecounts_initialization(mock_verify, mock_load_config, mock_which, mock_fc_config, temp_outdir):
    """Test FeatureCounts initialization."""
    mock_which.return_value = '/usr/bin/featureCounts'
    mock_load_config.return_value = mock_fc_config
    mock_verify.return_value = True

    quantifier = FeatureCounts(bam_dict={'sample1': ['/path/to/bam']}, annotation_file='/path/to/gtf', out_dir=temp_outdir)

    assert quantifier.tool_name == 'featurecounts' # Changed from name to tool_name
    assert quantifier.executable_path == '/usr/bin/featureCounts'
    assert quantifier.annotation_file == '/path/to/gtf'

@patch('shutil.which')
@patch('pyseqrna.modules.quantification.featurecounts.FeatureCountsQuantifier.load_config')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_build_command(mock_verify, mock_load_config, mock_which, mock_fc_config, temp_outdir):
    """Test FeatureCounts command preparation."""
    mock_which.return_value = '/usr/bin/featureCounts'
    mock_load_config.return_value = mock_fc_config
    mock_verify.return_value = True

    quantifier = FeatureCounts(bam_dict={'sample1': ['/path/to/bam']}, annotation_file='/path/to/gtf', out_dir=temp_outdir)

    # Test _build_command directly
    cmd = quantifier._build_command('sample1', '/path/to/sample1.bam')

    assert '/usr/bin/featureCounts' in cmd
    assert '-a /path/to/gtf' in cmd
    assert '-o' in cmd
    assert '/path/to/sample1.bam' in cmd

@patch('shutil.which')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_featurecounts_not_found(mock_verify, mock_which, temp_outdir):
    """Test error when FeatureCounts is not found."""
    mock_which.return_value = None
    mock_verify.return_value = True

    from pyseqrna.modules.quantification.base import QuantificationError
    with pytest.raises(QuantificationError, match="featureCounts executable not found"):
        FeatureCounts(bam_dict={'sample1': ['/path/to/bam']}, annotation_file='/path/to/gtf', out_dir=temp_outdir)
