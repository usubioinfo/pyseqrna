import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from pyseqrna.modules.coexpression import PyCoexpression, CoexpressionError


def test_pycoexpression_dryrun(temp_outdir):
    """Test PyCoexpression in dry-run mode."""
    matrix_file = os.path.join(temp_outdir, "dummy_matrix.csv")
    pd.DataFrame({"Gene": ["G1"], "S1": [1.0]}).to_csv(matrix_file, index=False)

    coexp = PyCoexpression(matrix_file=matrix_file, out_dir=temp_outdir, dryrun=True)

    assert coexp.dryrun is True
    res = coexp.run()
    assert res["command"] == "pycoexpression_native_run"
    assert res["out_dir"] == temp_outdir


def test_pycoexpression_load_matrix(temp_outdir):
    """Test loading different file formats."""
    # Test CSV
    csv_file = os.path.join(temp_outdir, "matrix.csv")
    df_data = pd.DataFrame({"Gene": ["G1", "G2"], "S1": [10, 20], "S2": [15, 25]})
    df_data.to_csv(csv_file, index=False)

    coexp_csv = PyCoexpression(matrix_file=csv_file, out_dir=temp_outdir)
    loaded_csv = coexp_csv._load_matrix()
    assert list(loaded_csv.columns) == ["Gene", "S1", "S2"]

    # Test TSV
    tsv_file = os.path.join(temp_outdir, "matrix.tsv")
    df_data.to_csv(tsv_file, sep="\t", index=False)
    coexp_tsv = PyCoexpression(matrix_file=tsv_file, out_dir=temp_outdir)
    loaded_tsv = coexp_tsv._load_matrix()
    assert list(loaded_tsv.columns) == ["Gene", "S1", "S2"]

    # Test unsupported extension
    bad_file = os.path.join(temp_outdir, "matrix.bad")
    with open(bad_file, "w") as f:
        f.write("foo")
    coexp_bad = PyCoexpression(matrix_file=bad_file, out_dir=temp_outdir)
    with pytest.raises(CoexpressionError, match="Unsupported matrix file format"):
        coexp_bad._load_matrix()


def test_pycoexpression_collapse_replicates(temp_outdir):
    """Test replicate averaging logic."""
    csv_file = os.path.join(temp_outdir, "matrix_reps.csv")
    # Col names matching the grouping regexes:
    # CondA_1, CondA_2
    # CondB-R1, CondB-R2
    # CondC1, CondC2
    # CondD (no suffix)
    df_data = pd.DataFrame(
        {
            "Gene": ["G1", "G2"],
            "CondA_1": [10.0, 20.0],
            "CondA_2": [20.0, 30.0],
            "CondB-R1": [30.0, 40.0],
            "CondB-R2": [40.0, 50.0],
            "CondC1": [50.0, 60.0],
            "CondC2": [60.0, 70.0],
            "CondD": [70.0, 80.0],
        }
    )
    df_data.to_csv(csv_file, index=False)

    coexp = PyCoexpression(matrix_file=csv_file, out_dir=temp_outdir)
    loaded = coexp._load_matrix()
    expr_df = loaded.drop(columns=["Gene"])

    collapsed = coexp._collapse_replicates(expr_df)

    # CondA should be average of CondA_1 (10, 20) and CondA_2 (20, 30) => (15, 25)
    assert list(collapsed.columns) == ["CondA", "CondB", "CondC", "CondD"]
    assert collapsed.loc[0, "CondA"] == 15.0
    assert collapsed.loc[1, "CondA"] == 25.0
    # CondB should be (35, 45)
    assert collapsed.loc[0, "CondB"] == 35.0
    # CondC should be (55, 65)
    assert collapsed.loc[0, "CondC"] == 55.0
    # CondD should be (70, 80)
    assert collapsed.loc[0, "CondD"] == 70.0


def test_pycoexpression_run_basic(temp_outdir):
    """Test PyCoexpression run with a small synthetic dataset."""
    # Let's generate 40 genes and 6 conditions.
    # Group 1: 15 genes with a pattern [1, 2, 3, 2, 1, 0] (plus noise)
    # Group 2: 15 genes with a pattern [3, 2, 1, 2, 3, 4] (plus noise)
    # Group 3: 10 flat genes (low variance) [1, 1, 1, 1, 1, 1] (variance < 0.1)

    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(40)]

    data = {"Gene": genes}
    samples = ["S1", "S2", "S3", "S4", "S5", "S6"]

    for s_idx, s in enumerate(samples):
        values = []
        for i in range(40):
            if i < 15:
                # Group 1 profile
                p = [1.0, 2.0, 3.0, 2.0, 1.0, 0.0][s_idx]
                val = p + np.random.normal(0, 0.1)
            elif i < 30:
                # Group 2 profile
                p = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0][s_idx]
                val = p + np.random.normal(0, 0.1)
            else:
                # Flat/low variance group
                val = 1.0 + np.random.normal(0, 0.01)
            values.append(val)
        data[s] = values

    df = pd.DataFrame(data)
    matrix_file = os.path.join(temp_outdir, "synthetic_matrix.csv")
    df.to_csv(matrix_file, index=False)

    coexp = PyCoexpression(matrix_file=matrix_file, out_dir=temp_outdir)

    # Run with K = 2 and small cluster_size = 5
    res = coexp.run(tightness=1.0, k_values="2", outlier=3.0, cluster_size=5, replicates=False, preprocessing=False)

    assert res["command"] == "pycoexpression_native_run"

    # Verify outputs are written
    out_path = Path(temp_outdir)
    assert (out_path / "coexpression_clusters.tsv").exists()
    assert (out_path / "Unclustered_genes.tsv").exists()

    # Read clusters mapping
    clusters_df = pd.read_csv(out_path / "coexpression_clusters.tsv", sep="\t")
    # Flat genes (index 30-39) should be filtered out / unclustered (-1)
    unclustered = clusters_df[clusters_df["Cluster"] == -1]["Gene"].tolist()
    for i in range(30, 40):
        assert f"Gene_{i}" in unclustered

    # Group 1 and 2 genes should be clustered into cluster 1 or 2 (valid clusters)
    valid_clustered = clusters_df[clusters_df["Cluster"].isin([1, 2])]
    assert len(valid_clustered) >= 25  # most should be clustered correctly

    # Verify cluster specific files exist
    assert (out_path / "Cluster_1.tsv").exists()
    assert (out_path / "Cluster_1.png").exists()
    assert (out_path / "Cluster_2.tsv").exists()
    assert (out_path / "Cluster_2.png").exists()


def test_pycoexpression_optimal_k_selection(temp_outdir):
    """Test Silhouette-based K estimation."""
    # Let's generate a dataset with K=3 distinct groups
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(45)]
    data = {"Gene": genes}
    samples = ["S1", "S2", "S3", "S4", "S5", "S6"]

    for s_idx, s in enumerate(samples):
        values = []
        for i in range(45):
            if i < 15:
                p = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0][s_idx]
            elif i < 30:
                p = [5.0, 1.0, 5.0, 1.0, 5.0, 1.0][s_idx]
            else:
                p = [3.0, 3.0, -1.0, -1.0, 5.0, 5.0][s_idx]
            values.append(p + np.random.normal(0, 0.1))
        data[s] = values

    df = pd.DataFrame(data)
    matrix_file = os.path.join(temp_outdir, "k_matrix.csv")
    df.to_csv(matrix_file, index=False)

    coexp = PyCoexpression(matrix_file=matrix_file, out_dir=temp_outdir)

    # Test range: 2 3 4
    # The silhouette optimizer should evaluate 2, 3, 4 and select the best (should be 3)
    res = coexp.run(tightness=1.0, k_values="2 3 4", outlier=3.0, cluster_size=5, replicates=False, preprocessing=False)

    out_path = Path(temp_outdir)
    assert (out_path / "coexpression_clusters.tsv").exists()
    # Check that Cluster 3 files exist
    assert (out_path / "Cluster_3.tsv").exists()


def test_pycoexpression_errors(temp_outdir):
    """Test exceptions handled by PyCoexpression."""
    # Non-existent file
    coexp_missing = PyCoexpression(matrix_file="missing.csv", out_dir=temp_outdir)
    with pytest.raises(CoexpressionError, match="Expression matrix not found"):
        coexp_missing.run()

    # Empty file
    empty_file = os.path.join(temp_outdir, "empty.csv")
    pd.DataFrame().to_csv(empty_file, index=False)
    coexp_empty = PyCoexpression(matrix_file=empty_file, out_dir=temp_outdir)
    with pytest.raises(CoexpressionError, match="Failed to read expression matrix"):
        coexp_empty.run()

    # Not enough non-flat genes
    flat_file = os.path.join(temp_outdir, "flat.csv")
    pd.DataFrame({"Gene": ["G1", "G2"], "S1": [1.0, 1.0], "S2": [1.0, 1.0]}).to_csv(flat_file, index=False)
    coexp_flat = PyCoexpression(matrix_file=flat_file, out_dir=temp_outdir)
    with pytest.raises(CoexpressionError, match="Not enough non-flat genes available for clustering"):
        coexp_flat.run()
