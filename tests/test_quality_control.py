import pytest
import os
from unittest.mock import patch, MagicMock
from pyseqrna.modules.quality.fastqc import FastQCQualityControl


@pytest.fixture
def mock_fastqc_config():
    return {"fastqc": {"--threads": "4", "--noextract": ""}}


@patch("shutil.which")
@patch("pyseqrna.modules.quality.fastqc.FastQCQualityControl.load_config")
@patch("pyseqrna.utils.file_manager.FileManager.verify_files_exist")
@patch("os.getcwd")
def test_fastqc_initialization(mock_getcwd, mock_verify, mock_load_config, mock_which, mock_fastqc_config, temp_outdir):
    """Test FastQC initialization."""
    mock_which.return_value = "/usr/bin/fastqc"
    mock_load_config.return_value = mock_fastqc_config
    mock_verify.return_value = True
    mock_getcwd.return_value = os.path.dirname(temp_outdir)

    qc = FastQCQualityControl(sample_dict={"sample1": ["/path/to/read1.fq"]}, out_dir=temp_outdir, logger=None)

    assert qc.name == "fastqc"


@patch("shutil.which")
@patch("pyseqrna.modules.quality.fastqc.FastQCQualityControl.load_config")
@patch("pyseqrna.utils.file_manager.FileManager.verify_files_exist")
@patch("os.getcwd")
def test_prepare_command(mock_getcwd, mock_verify, mock_load_config, mock_which, mock_fastqc_config, temp_outdir):
    """Test FastQC command preparation."""
    mock_which.return_value = "/usr/bin/fastqc"
    mock_load_config.return_value = mock_fastqc_config
    mock_verify.return_value = True
    mock_getcwd.return_value = os.path.dirname(temp_outdir)

    # Added cpu_threads=4 to match expected output
    qc = FastQCQualityControl(sample_dict={"sample1": ["/path/to/read1.fq"]}, out_dir=temp_outdir, logger=None, cpu_threads=4)

    reads = ["sample_R1.fastq.gz", "sample_R2.fastq.gz"]
    cmd = qc.prepare_command("sample1", reads)

    assert "fastqc" in cmd
    assert "-o" in cmd
    assert "sample_R1.fastq.gz" in cmd
    assert "sample_R2.fastq.gz" in cmd
    assert "--threads 4" in cmd


@patch("shutil.which")
@patch("pyseqrna.utils.file_manager.FileManager.verify_files_exist")
@patch("os.getcwd")
def test_fastqc_not_found(mock_getcwd, mock_verify, mock_which, temp_outdir):
    """Test error when FastQC is not found."""
    mock_which.return_value = None
    mock_verify.return_value = True
    mock_getcwd.return_value = os.path.dirname(temp_outdir)

    from pyseqrna.modules.quality.base import QualityControlError

    with pytest.raises(QualityControlError, match="FastQC executable not found"):
        FastQCQualityControl(sample_dict={"sample1": ["/path/to/read1.fq"]}, out_dir=temp_outdir, logger=None)
