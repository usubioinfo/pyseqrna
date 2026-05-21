import pytest
import os
import json
from pyseqrna.pipeline.pipeline import Pipeline
from pyseqrna.utils.checkpoint_manager import CheckpointManager
from pyseqrna.utils.file_manager import FileManager
from pyseqrna.utils.stage_registry import PIPELINE_STAGE_NAMES, get_pipeline_stages
from pyseqrna.__version__ import __version__
from pyseqrna.cli.argument_manager import ArgumentManager, __version__ as cli_version

def test_pipeline_init(temp_outdir):
    """Test Pipeline initialization."""
    # Create a dummy config file
    config_file = os.path.join(temp_outdir, "config.ini")
    with open(config_file, 'w') as f:
        f.write("[general]\nproject_name=test_project\n")

    try:
        pipeline = Pipeline(
            input_file='dummy_input.txt',
            samples_path='dummy_samples',
            reference_genome='dummy_genome.fasta',
            feature_file='dummy_annotation.gtf',
            output_dir=temp_outdir,
            logger=None
        )

        assert pipeline.output_dir == os.path.abspath(temp_outdir)
        # assert "visualization" in pipeline.pipeline_stages # Removed invalid assertion
        assert hasattr(pipeline, 'run_visualization') # Check if method exists

    except Exception as e:
        pytest.fail(f"Pipeline initialization failed: {e}")

def test_pipeline_stages_order(temp_outdir):
    """Test that all expected pipeline stage methods exist in the correct logical order."""
    # Verify all expected stage methods exist on the Pipeline class
    expected_stages = [
        'run_quality_control',
        'run_trimming',
        'run_quality_control_trim',
        'run_alignment',
        'run_quantification',
        'run_normalization',
        'run_differential_expression',
        'run_visualization',
        'run_gene_ontology',
        'run_pathway_enrichment',
    ]

    for stage in expected_stages:
        assert hasattr(Pipeline, stage), f"Pipeline missing expected stage method: {stage}"
        assert callable(getattr(Pipeline, stage)), f"Pipeline.{stage} is not callable"


def test_stage_registry_matches_pipeline(temp_outdir):
    """Test that shared stage metadata can build the pipeline execution plan."""
    pipeline = Pipeline(
        input_file='dummy_input.txt',
        samples_path='dummy_samples',
        reference_genome='dummy_genome.fasta',
        feature_file='dummy_annotation.gtf',
        output_dir=temp_outdir,
        logger=None
    )

    stages = get_pipeline_stages(pipeline)

    assert [stage for stage, _ in stages] == list(PIPELINE_STAGE_NAMES)
    assert all(callable(callback) for _, callback in stages)


def test_checkpoint_stage_order_uses_registry(temp_outdir):
    """Test that checkpoint stage ordering follows production pipeline order."""
    manager = CheckpointManager(temp_outdir)

    assert manager.get_incomplete_stages()[:4] == [
        "quality",
        "trimming",
        "quality_trim",
        "alignment",
    ]
    assert manager.get_stage_dependencies("trimming") == ["quality"]
    assert manager.get_stage_dependencies("differential") == ["quantification"]


def test_resume_policy_is_deterministic(temp_outdir):
    """Test completed-stage handling without interactive prompts."""
    pipeline = Pipeline(
        input_file='dummy_input.txt',
        samples_path='dummy_samples',
        reference_genome='dummy_genome.fasta',
        feature_file='dummy_annotation.gtf',
        output_dir=temp_outdir,
        logger=None,
        resume_policy="skip",
    )

    assert pipeline._ask_user_rerun_stage("quality", "fastqc") is False

    pipeline.resume_policy = "rerun"
    assert pipeline._ask_user_rerun_stage("quality", "fastqc") is True

    pipeline.resume_policy = "fail"
    with pytest.raises(ValueError, match="already complete"):
        pipeline._ask_user_rerun_stage("quality", "fastqc")

    pipeline.resume_policy = "prompt"
    pipeline._can_prompt_user = lambda: False
    with pytest.raises(ValueError, match="non-interactive"):
        pipeline._ask_user_rerun_stage("quality", "fastqc")


def test_run_record_contains_resume_policy(temp_outdir):
    """Test that run records capture deterministic resume behavior."""
    pipeline = Pipeline(
        input_file='dummy_input.txt',
        samples_path='dummy_samples',
        reference_genome='dummy_genome.fasta',
        feature_file='dummy_annotation.gtf',
        output_dir=temp_outdir,
        logger=None,
        resume_policy="rerun",
    )

    record_path = pipeline._write_run_record(success=True)
    payload = json.loads(record_path.read_text())

    assert payload["pyseqrna_version"] == __version__
    assert payload["execution"]["resume_policy"] == "rerun"


def test_cli_version_uses_package_version():
    """Test that CLI version metadata cannot drift from package version."""
    assert cli_version == __version__


def test_config_rejects_unknown_key(tmp_path):
    """Test that production config typos fail fast."""
    config_file = tmp_path / "bad.ini"
    config_file.write_text(
        "[General]\n"
        "input_file = samples.tsv\n"
        "samples_path = reads\n"
        "reference_genome = ref.fa\n"
        "feature_file = genes.gtf\n"
        "slurm_memory_per_taks = 64\n",
        encoding="utf-8",
    )

    manager = ArgumentManager()

    with pytest.raises(SystemExit, match="Unknown config key"):
        manager._read_run_config(str(config_file))


def test_config_rejects_bad_choice(tmp_path):
    """Test that INI values obey the same choices as CLI arguments."""
    config_file = tmp_path / "bad_choice.ini"
    config_file.write_text(
        "[General]\n"
        "input_file = samples.tsv\n"
        "samples_path = reads\n"
        "reference_genome = ref.fa\n"
        "feature_file = genes.gtf\n"
        "resume_policy = skipp\n",
        encoding="utf-8",
    )

    manager = ArgumentManager()

    with pytest.raises(SystemExit, match="Invalid value"):
        manager._read_run_config(str(config_file))


def test_config_choices_are_case_insensitive(tmp_path):
    """Test that strict INI validation still accepts common case variants."""
    config_file = tmp_path / "case.ini"
    config_file.write_text(
        "[General]\n"
        "input_file = samples.tsv\n"
        "samples_path = reads\n"
        "reference_genome = ref.fa\n"
        "feature_file = genes.gtf\n"
        "alignment_tool = STAR\n"
        "source = ensembl\n",
        encoding="utf-8",
    )

    manager = ArgumentManager()
    defaults = manager._read_run_config(str(config_file))

    assert defaults["alignment_tool"] == "star"
    assert defaults["source"] == "ENSEMBL"


def test_config_accepts_slurm_wait_timeout(tmp_path):
    """Test that SLURM timeout is configurable from production INI files."""
    config_file = tmp_path / "slurm_timeout.ini"
    config_file.write_text(
        "[General]\n"
        "input_file = samples.tsv\n"
        "samples_path = reads\n"
        "reference_genome = ref.fa\n"
        "feature_file = genes.gtf\n"
        "slurm_wait_timeout_hours = 12.5\n",
        encoding="utf-8",
    )

    manager = ArgumentManager()
    defaults = manager._read_run_config(str(config_file))

    assert defaults["slurm_wait_timeout_hours"] == 12.5


def test_config_rejects_bad_boolean():
    """Test that ambiguous boolean strings are not silently treated as False."""
    manager = ArgumentManager()

    with pytest.raises(SystemExit, match="Invalid boolean"):
        manager._parse_bool("maybe", key="dryrun", section="General", config_path="input_file.ini")


def test_existing_output_dir_fails_non_interactive(temp_outdir):
    """Test that batch runs do not block on output-directory prompts."""
    file_manager = FileManager(logger=None)

    with pytest.raises(FileExistsError, match="Non-interactive"):
        file_manager.create_main_output_directory(temp_outdir, allow_prompt=False)
