from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Settings
from schemas.models import Paper


@pytest.fixture
def settings_factory(tmp_path):
    def build(**overrides):
        values = {
            "provider": "template",
            "data_dir": tmp_path,
            "cache_ttl_hours": 0,
            "request_timeout": 5,
            "max_context_tokens": 4096,
            "max_output_tokens": 512,
            "max_retries": 0,
        }
        values.update(overrides)
        return Settings(**values)
    return build


@pytest.fixture
def real_paper():
    return Paper(
        paper_id="arxiv:1101.5027",
        title="Modeling and Solving AP Location and Frequency Assignment for Maximizing Access Efficiency in Wi-Fi Networks",
        authors=[],
        abstract=(
            "In this paper, we present optimization approaches for Access Point (AP) location and frequency assignment, "
            "two major planning tasks in deploying Wi-Fi networks. We consider a performance metric, referred to as "
            "access efficiency, that captures key aspects of how user devices share access to the wireless medium."
        ),
        year=2011,
        url="https://arxiv.org/abs/1101.5027",
        pdf_url=None,
        source="arXiv",
        source_ids=["arXiv:1101.5027"],
        is_preprint=True,
    )
