import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyseqrna.pipeline.pipeline import Pipeline, _RawLoggerAdapter
from pyseqrna.utils.command_executor import CommandExecutor
from pyseqrna.utils.config_manager import ConfigManager
from pyseqrna.utils.resource_manager import ResourceManager


def test_config_manager_missing_file_raises_file_not_found(tmp_path):
    manager = ConfigManager(logging.getLogger("test_config_missing"))

    with pytest.raises(FileNotFoundError):
        manager.read_runconfig(str(tmp_path / "missing.ini"))


def test_config_manager_expands_user_paths(monkeypatch, tmp_path):
    manager = ConfigManager(logging.getLogger("test_config_paths"))
    fake_home = tmp_path / "home"
    ref_dir = fake_home / "refs"
    ref_dir.mkdir(parents=True)
    (ref_dir / "genome.fa").write_text(">chr1\nACGT\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    manager.validated_config = {"General": {"reference_genome": "~/refs/../refs/genome.fa"}}

    path = manager.get_path("General", "reference_genome")

    assert isinstance(path, Path)
    assert "~" not in str(path)
    assert ".." not in str(path)


def test_resource_manager_writes_slurm_ini_to_output_dir(tmp_path):
    manager = ResourceManager(logging.getLogger("test_resource_slurm_ini"))

    config_path = Path(
        manager.write_slurm_ini(
            partition="compute",
            cpus=4,
            memory=32,
            ntasks=1,
            output_dir=str(tmp_path),
        )
    )

    assert config_path == tmp_path / "slurm.ini"
    assert config_path.exists()


def test_command_executor_local_uses_shell_free_execution(tmp_path):
    executor = CommandExecutor(logging.getLogger("test_command_executor"))
    command = f"{sys.executable} -c 'print(\"shell free ok\")'"

    result = executor.execute_local({"sample1": command}, "dummy", str(tmp_path))

    assert result == {"sample1": True}
    assert "shell free ok" in (tmp_path / "logs" / "sample1_dummy.out").read_text()


def test_command_executor_local_supports_pipelines_without_shell_true(tmp_path):
    executor = CommandExecutor(logging.getLogger("test_command_executor_pipeline"))
    command = "printf 'pipe ok' | wc -c"

    result = executor.execute_local({"sample1": command}, "dummy", str(tmp_path))

    assert result == {"sample1": True}
    assert "7" in (tmp_path / "logs" / "sample1_dummy.out").read_text()


def test_command_executor_rejects_invalid_slurm_wait_timeout():
    with pytest.raises(ValueError, match="slurm_wait_timeout_seconds"):
        CommandExecutor(logging.getLogger("test_command_executor_timeout"), slurm_wait_timeout_seconds=0)


def test_command_executor_uses_slurm_config_wait_timeout():
    executor = CommandExecutor(logging.getLogger("test_command_executor_timeout"), slurm_wait_timeout_seconds=10)

    timeout = executor._slurm_wait_timeout({"wait_timeout_seconds": "30"})

    assert timeout == 30


def test_slurm_script_avoids_eval_for_tool_commands(tmp_path):
    executor = CommandExecutor(logging.getLogger("test_command_executor_slurm_script"))
    command = f"{sys.executable} -c 'print(\"safe slurm\")'"

    script = Path(
        executor._create_slurm_script(
            "sample1",
            "dummy",
            command,
            tmp_path / "logs",
            {"partition": "compute"},
        )
    )

    text = script.read_text()
    assert 'eval "$COMMAND"' not in text
    assert 'PYSEQRNA_COMMAND="$COMMAND"' in text


def test_raw_logger_adapter_forwards_exception():
    raw = MagicMock(spec=logging.Logger)
    adapter = _RawLoggerAdapter(raw)

    adapter.exception("boom %s", "now")

    raw.exception.assert_called_once_with("boom %s", "now")


def test_pipeline_rejects_invalid_local_jobs(temp_outdir):
    with pytest.raises(ValueError, match="local_jobs"):
        Pipeline(
            input_file="samples.tsv",
            samples_path="reads",
            reference_genome="reference.fa",
            feature_file="genes.gtf",
            output_dir=temp_outdir,
            local_jobs=0,
        )


def test_pipeline_rejects_invalid_slurm_wait_timeout(temp_outdir):
    with pytest.raises(ValueError, match="slurm_wait_timeout_hours"):
        Pipeline(
            input_file="samples.tsv",
            samples_path="reads",
            reference_genome="reference.fa",
            feature_file="genes.gtf",
            output_dir=temp_outdir,
            slurm_wait_timeout_hours=0,
        )


def test_pipeline_slurm_config_contains_wait_timeout(temp_outdir):
    pipeline = Pipeline(
        input_file="samples.tsv",
        samples_path="reads",
        reference_genome="reference.fa",
        feature_file="genes.gtf",
        output_dir=temp_outdir,
        slurm_wait_timeout_hours=1.5,
    )

    assert pipeline._get_slurm_config()["wait_timeout_seconds"] == "5400"


def test_pipeline_accepts_raw_logger(temp_outdir):
    raw_logger = logging.getLogger("test_pipeline_raw_logger")

    pipeline = Pipeline(
        input_file="samples.tsv",
        samples_path="reads",
        reference_genome="reference.fa",
        feature_file="genes.gtf",
        output_dir=temp_outdir,
        logger=raw_logger,
    )

    assert isinstance(pipeline.logger, _RawLoggerAdapter)
