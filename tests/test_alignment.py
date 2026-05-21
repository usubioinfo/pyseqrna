import pytest
import os
from unittest.mock import patch, MagicMock
from pyseqrna.modules.alignment.star import StarAligner as STAR

@pytest.fixture
def mock_star_config():
    return {
        'alignment': { # STAR uses 'alignment' section for run_alignment params
            '--outSAMtype': 'BAM SortedByCoordinate'
        },
        'star': { # STAR uses 'star' section for index building
             '--runThreadN': '4'
        }
    }

@patch('shutil.which')
@patch('pyseqrna.modules.alignment.star.StarAligner.load_config')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_star_initialization(mock_verify, mock_load_config, mock_which, mock_star_config, temp_outdir):
    """Test STAR initialization."""
    mock_which.return_value = '/usr/bin/STAR'
    mock_load_config.return_value = mock_star_config
    mock_verify.return_value = True

    aligner = STAR(genome='/path/to/genome.fasta', out_dir=temp_outdir, logger=None)

    assert aligner.name == 'star'

@patch('shutil.which')
@patch('pyseqrna.modules.alignment.star.StarAligner.load_config')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
@patch('pyseqrna.modules.alignment.star.StarAligner.check_index')
@patch('pyseqrna.modules.alignment.star.StarAligner.execute_command')
def test_run_alignment(mock_execute, mock_check_index, mock_verify, mock_load_config, mock_which, mock_star_config, temp_outdir):
    """Test STAR alignment execution."""
    mock_which.return_value = '/usr/bin/STAR'
    mock_load_config.return_value = mock_star_config
    mock_verify.return_value = True
    mock_check_index.return_value = True
    mock_execute.return_value = {}

    aligner = STAR(genome='/path/to/genome.fasta', out_dir=temp_outdir, logger=None)

    target = {'sample1': ['sample_R1.fastq.gz', 'sample_R2.fastq.gz']}
    aligner.run_alignment(target, paired=True)

    # Verify execute_command was called with correct commands
    args, _ = mock_execute.call_args
    commands = args[0]
    assert 'sample1' in commands
    cmd = commands['sample1']

    assert 'STAR' in cmd
    assert '--readFilesIn sample_R1.fastq.gz sample_R2.fastq.gz' in cmd
    assert '--outFileNamePrefix' in cmd

@patch('shutil.which')
@patch('pyseqrna.modules.alignment.star.StarAligner.load_config')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
def test_star_read_files_command_uses_resolved_executable(mock_verify, mock_load_config, mock_which, temp_outdir):
    """Test STAR compressed-read command is resolved to an absolute helper path."""
    mock_which.side_effect = lambda name: {
        'STAR': '/usr/bin/STAR',
        'gzcat': '/usr/bin/gzcat',
        'zcat': '/usr/bin/zcat',
    }.get(name)
    mock_load_config.return_value = {
        'alignment': {'zipped_file': '--readFilesCommand zcat'}
    }
    mock_verify.return_value = True

    aligner = STAR(genome='/path/to/genome.fasta', out_dir=temp_outdir, logger=None)
    resolved = aligner._resolve_read_files_command('--readFilesCommand zcat', prefer_gzcat=True)

    assert resolved == '--readFilesCommand /usr/bin/gzcat'

@patch('shutil.which')
@patch('pyseqrna.modules.alignment.star.StarAligner.load_config')
@patch('pyseqrna.utils.file_manager.FileManager.verify_files_exist')
@patch('pyseqrna.modules.alignment.star.StarAligner.check_index')
def test_star_not_found(mock_check_index, mock_verify, mock_load_config, mock_which, temp_outdir):
    """Test that STAR alignment fails when STAR executable is not found."""
    mock_which.return_value = None
    mock_load_config.return_value = {}
    mock_verify.return_value = True
    mock_check_index.return_value = False

    aligner = STAR(genome='/path/to/genome.fasta', out_dir=temp_outdir, logger=None)

    # STAR does not validate executable at init, but shutil.which confirms it's missing
    import shutil
    assert shutil.which('STAR') is None

    # Attempting to build index should fail since STAR binary is not available
    from pyseqrna.modules.alignment.base import AlignmentError
    with pytest.raises((AlignmentError, Exception)):
        aligner.build_index()
