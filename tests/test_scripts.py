from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    merged["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / script)],
        cwd=ROOT,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


def test_caseops_builder_does_not_require_the_old_home_checkout(tmp_path: Path):
    result = _run(
        "build_sp10240_caseops_local.sh",
        {"DOCS_PATH": str(tmp_path / "missing.jsonl"), "OUT_ROOT": str(tmp_path / "out")},
    )
    assert result.returncode == 1
    assert "docs not found" in result.stderr
    assert "sota_rascal" not in result.stderr


def test_stream_script_reports_a_missing_docs_file(tmp_path: Path):
    result = _run("stream_pr1855_caseops_to_pod.sh", {"DOCS": str(tmp_path / "missing.jsonl")})
    assert result.returncode == 1
    assert "docs not found" in result.stderr


def test_upload_check_reports_counts_before_asking_for_hf(tmp_path: Path):
    out = tmp_path / "datasets"
    dataset = out / "datasets" / "fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
    tokenizers = out / "tokenizers"
    dataset.mkdir(parents=True)
    tokenizers.mkdir()
    (tokenizers / "fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model").write_bytes(b"model")
    (tokenizers / "fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.vocab").write_bytes(b"vocab")
    (out / "caseops_manifest.json").write_text("{}\n")
    (dataset / "fineweb_train_000000.bin").write_bytes(b"train")
    (dataset / "fineweb_val_000000.bin").write_bytes(b"val")
    (dataset / "fineweb_val_bytes_000000.bin").write_bytes(b"bytes")
    result = _run(
        "upload_sp10240_caseops_to_hf.sh",
        {"OUT_ROOT": str(out), "HF_BIN": str(tmp_path / "missing-hf"), "PYTHON_BIN": "python3"},
    )
    assert result.returncode == 1
    assert "train_shards=1 val_shards=1 val_byte_sidecars=1" in result.stdout
    assert "val_shards=2" not in result.stdout
    assert "hf CLI not found" in result.stderr


def test_upload_check_stops_on_missing_dataset_without_hf(tmp_path: Path):
    result = _run(
        "upload_sp10240_caseops_to_hf.sh",
        {"OUT_ROOT": str(tmp_path / "empty"), "HF_BIN": str(tmp_path / "missing-hf")},
    )
    assert result.returncode == 2
    assert "MISSING:" in result.stderr
    assert "not upload-ready" in result.stderr
    assert "hf CLI" not in result.stderr


def test_queue_script_reports_a_missing_runner(tmp_path: Path):
    result = _run(
        "queue_10k_after_current_pr1855.sh",
        {"TENK_DIR": str(tmp_path / "missing-leg"), "LOG_DIR": str(tmp_path / "logs")},
    )
    assert result.returncode == 1
    assert "10k runner dir not found" in result.stderr
    assert not (tmp_path / "logs").exists()


def test_sp5200_builder_calls_the_repo_validator():
    text = (ROOT / "scripts" / "build_sp5200_dataset.sh").read_text()
    assert "/workspace/sota_rascal/scripts/validate_fineweb_shards.py" not in text
    assert "validate_fineweb_shards.py" in text
    assert 'SPEC="${SPEC:-$ROOT/scripts/tokenizer_specs_sp5200.json}"' in text
