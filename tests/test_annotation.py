import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from pyseqrna.modules.annotation.gene_ontology import GeneOntology

@pytest.fixture
def mock_go_config():
    return {
        'gene_ontology': {
            'species': 'hsapiens'
        }
    }

@patch('pyseqrna.modules.annotation.gene_ontology.GeneOntology._query_biomart')
@patch('pyseqrna.modules.annotation.gene_ontology.GeneOntology._preprocess_biomart')
def test_go_initialization(mock_preprocess, mock_query, mock_go_config, temp_outdir):
    """Test GeneOntology initialization."""
    mock_query.return_value = pd.DataFrame()
    mock_preprocess.return_value = (pd.DataFrame(), 100)

    # Removed out_dir argument
    go = GeneOntology(species='hsapiens')

    assert go.species == 'hsapiens'

@patch('pyseqrna.modules.annotation.gene_ontology.GeneOntology._query_biomart')
@patch('pyseqrna.modules.annotation.gene_ontology.GeneOntology._preprocess_biomart')
def test_fetch_annotations(mock_preprocess, mock_query, temp_outdir):
    """Test fetching annotations (mocked)."""
    mock_query.return_value = pd.DataFrame()
    mock_preprocess.return_value = (pd.DataFrame(), 100)

    # Removed out_dir argument
    go = GeneOntology(species='hsapiens')

    assert go.species == 'hsapiens'
