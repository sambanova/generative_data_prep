import os
from pathlib import Path

import pytest

from generative_data_prep.utils import validate_sha256


def test_validation_sha_with_split():
    """Testing validation method and sha256 created correctly. With splits."""
    validate_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_split", "pipelined_pretraining_sha256_split"
    )
    # breakpoint()
    assert validate_sha256(validate_dir)


def fake_getsize(file_path):
    """Fake get size function for monkeypatch."""
    return 0


@pytest.fixture
def mock_getsize(monkeypatch):
    """Mocking size function using monkeypatch."""
    monkeypatch.setattr(os.path, "getsize", fake_getsize)


def test_validation_sha_with_split_redoing_sha256(mock_getsize):
    """Testing validation method and sha256 created correctly. With splits."""
    validate_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_split", "pipelined_pretraining_sha256_split"
    )
    # breakpoint()
    assert validate_sha256(validate_dir)


def test_validation_sha_with_split_and_eval():
    """Testing validation method and sha256 created correctly. With splits."""
    validate_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_sha256_split_and_eval",
        "pipelined_pretraining_sha256_split_and_eval",
    )
    # breakpoint()
    assert validate_sha256(validate_dir)


def test_validation_sha_without_split():
    """Testing validation method and sha256 created correctly. Without splits."""
    validate_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_no_split", "pipelined_pretraining_sha256_no_split"
    )
    assert validate_sha256(validate_dir)
