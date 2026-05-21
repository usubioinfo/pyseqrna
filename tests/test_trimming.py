import pytest
import os
from unittest.mock import patch, MagicMock
from pyseqrna.modules.trimming.trim_galore import TrimGaloreTrimmer as TrimGalore


@pytest.fixture
def mock_trim_config():
    return {"trim_galore": {"--quality": "20", "--length": "20"}}


@patch("shutil.which")
@patch("pyseqrna.modules.trimming.trim_galore.TrimGaloreTrimmer.load_config")
@patch("pyseqrna.utils.file_manager.FileManager.verify_files_exist")
def test_trimgalore_initialization(mock_verify, mock_load_config, mock_which, mock_trim_config, temp_outdir):
    """Test TrimGalore initialization."""
    mock_which.return_value = "/usr/bin/trim_galore"
    mock_load_config.return_value = mock_trim_config
    mock_verify.return_value = True

    trimmer = TrimGalore(sample_dict={"sample1": ["/path/to/read1.fq"]}, out_dir=temp_outdir, logger=None)

    assert trimmer.name == "trim_galore"


@patch("shutil.which")
@patch("pyseqrna.modules.trimming.trim_galore.TrimGaloreTrimmer.load_config")
@patch("pyseqrna.utils.file_manager.FileManager.verify_files_exist")
def test_prepare_command(mock_verify, mock_load_config, mock_which, mock_trim_config, temp_outdir):
    """Test TrimGalore command preparation."""
    mock_which.return_value = "/usr/bin/trim_galore"
    mock_load_config.return_value = mock_trim_config
    mock_verify.return_value = True

    # Added paired=True
    trimmer = TrimGalore(sample_dict={"sample1": ["/path/to/read1.fq"]}, out_dir=temp_outdir, logger=None, paired=True)

    reads = ["sample_R1.fastq.gz", "sample_R2.fastq.gz"]
    cmd = trimmer.prepare_command("sample1", reads)

    assert "trimgalore" in cmd or "trim_galore" in cmd
    assert "-o" in cmd or "--output_dir" in cmd
    assert "sample_R1.fastq.gz" in cmd
    assert "sample_R2.fastq.gz" in cmd
    assert "--paired" in cmd
    assert "--basename sample1" in cmd


@patch("shutil.which")
@patch("pyseqrna.utils.file_manager.FileManager.verify_files_exist")
def test_trimgalore_not_found(mock_verify, mock_which, temp_outdir):
    """Test error when TrimGalore is not found."""
    mock_which.return_value = None
    mock_verify.return_value = True

    from pyseqrna.modules.trimming.base import TrimmingError

    with pytest.raises(TrimmingError, match="Trim Galore executable not found"):
        TrimGalore(sample_dict={"sample1": ["/path/to/read1.fq"]}, out_dir=temp_outdir, logger=None)
