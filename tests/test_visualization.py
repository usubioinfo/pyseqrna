import pytest
import os
import pandas as pd
from pyseqrna.modules.visualization import Visualization

def test_visualization_init(temp_outdir):
    """Test initialization of Visualization class."""
    viz = Visualization(outdir=temp_outdir)
    assert viz.outdir == temp_outdir
    assert os.path.exists(temp_outdir)

def test_plot_volcano(temp_outdir, sample_deg_df):
    """Test Volcano plot generation."""
    viz = Visualization(outdir=temp_outdir)

    # Test with DataFrame input
    viz.plot_volcano(degDF=sample_deg_df, comp="CondA-CondB", prefix="test_")

    expected_file = os.path.join(temp_outdir, "Volcano_Plots", "test_CondA-CondB_volcano.png")
    assert os.path.exists(expected_file)
    assert os.path.exists(expected_file.replace(".png", ".pdf"))

def test_plot_ma(temp_outdir, sample_deg_df, sample_counts_df):
    """Test MA plot generation."""
    viz = Visualization(outdir=temp_outdir)

    viz.plot_ma(degDF=sample_deg_df, countDF=sample_counts_df, comp="CondA-CondB", prefix="test_")

    expected_file = os.path.join(temp_outdir, "MA_Plots", "test_CondA-CondB_ma.png")
    assert os.path.exists(expected_file)
    assert os.path.exists(expected_file.replace(".png", ".pdf"))

def test_pca_plot(temp_outdir, sample_counts_df):
    """Test PCA plot generation."""
    viz = Visualization(outdir=temp_outdir)

    viz.pca_plot(ncountdf=sample_counts_df, prefix="test_")

    expected_file = os.path.join(temp_outdir, "Sample_Plots", "test_PCA_plot.png")
    assert os.path.exists(expected_file)
    assert os.path.exists(expected_file.replace(".png", ".pdf"))

def test_plot_venn_and_upset(temp_outdir):
    """Test Venn and UpSet-style intersection plot generation."""
    viz = Visualization(outdir=temp_outdir)
    deg_file = os.path.join(temp_outdir, "Filtered_DEGs.xlsx")

    with pd.ExcelWriter(deg_file) as writer:
        pd.DataFrame({"Gene": ["GeneA", "GeneB", "GeneC"], "logFC": [2.0, -2.0, 1.5]}).to_excel(
            writer, sheet_name="CondA-CondB", index=False
        )
        pd.DataFrame({"Gene": ["GeneB", "GeneC", "GeneD"], "logFC": [-1.5, 2.2, -2.5]}).to_excel(
            writer, sheet_name="CondA-CondC", index=False
        )

    venn_files = viz.plot_venn(deg_file=deg_file)
    upset_file = viz.plot_upset(deg_file=deg_file)

    assert len(venn_files) == 1
    assert os.path.exists(venn_files[0])
    assert os.path.exists(venn_files[0].replace(".png", ".pdf"))
    assert upset_file is not None
    assert os.path.exists(upset_file)
    assert os.path.exists(upset_file.replace(".png", ".pdf"))

def test_dryrun(temp_outdir, sample_deg_df):
    """Test dry run mode."""
    viz = Visualization(outdir=temp_outdir, dryrun=True)

    viz.plot_volcano(degDF=sample_deg_df, comp="CondA-CondB", prefix="dryrun_")

    expected_file = os.path.join(temp_outdir, "Volcano_Plots", "dryrun_CondA-CondB_volcano.png")
    assert not os.path.exists(expected_file)

def test_run_method(temp_outdir, sample_deg_df, sample_counts_df):
    """Test the run method."""
    viz = Visualization(outdir=temp_outdir)

    # Save sample data to files
    deg_file = os.path.join(temp_outdir, "deg.xlsx")
    counts_file = os.path.join(temp_outdir, "counts.csv")

    sample_deg_df.to_excel(deg_file, index=False)
    sample_counts_df.to_csv(counts_file, index=False)

    viz.run(norm_counts_file=counts_file, de_results_file=deg_file)

    # Check if files were created
    assert os.path.exists(os.path.join(temp_outdir, "Volcano_Plots", "CondA-CondB_volcano.png"))
    assert os.path.exists(os.path.join(temp_outdir, "MA_Plots", "CondA-CondB_ma.png"))
    assert os.path.exists(os.path.join(temp_outdir, "Sample_Plots", "All_Samples_PCA_plot.png"))
