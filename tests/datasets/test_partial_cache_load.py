#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`try_load` must report an insufficient cache rather than raising.

Downloads are episode-selective (`LeRobotDataset._download` passes `allow_patterns`), so after
loading one episode subset the cache holds only that subset's parquet files. Loading a second,
disjoint subset then finds parquet files on disk whose rows the episode filter rejects
entirely, and `Dataset.from_parquet` raises `ValueError: Instruction "train" corresponds to no
data!`. That must be treated as "cache insufficient, go download", not propagated to the
caller — federated training loads one disjoint shard per client and hits it every time.
"""

from unittest.mock import patch

import pytest

from lerobot.datasets.dataset_reader import DatasetReader


class _Meta:
    """The slice of LeRobotDatasetMetadata that DatasetReader touches during a load."""

    features: dict = {}
    total_frames = 0
    total_episodes = 0
    depth_keys: list = []
    image_keys: list = []
    video_keys: list = []


def _reader(episodes):
    reader = DatasetReader.__new__(DatasetReader)
    reader._meta = _Meta()
    reader.episodes = episodes
    reader.hf_dataset = None
    return reader


@pytest.mark.parametrize(
    "error",
    [
        # nothing cached at all
        FileNotFoundError("Provided directory does not contain any parquet file: /cache/data"),
        NotADirectoryError("/cache/data"),
        # parquet files cached, but none holding the requested episodes
        ValueError('Instruction "train" corresponds to no data!'),
    ],
)
def test_try_load_reports_insufficient_cache_instead_of_raising(error):
    reader = _reader(episodes=[917, 918, 919])

    with patch.object(DatasetReader, "_load_hf_dataset", side_effect=error):
        assert reader.try_load() is False

    assert reader.hf_dataset is None


def test_try_load_succeeds_when_the_cache_covers_the_episodes():
    reader = _reader(episodes=[900, 901])
    sentinel = object()

    with (
        patch.object(DatasetReader, "_load_hf_dataset", return_value=sentinel),
        patch.object(DatasetReader, "_check_cached_episodes_sufficient", return_value=True),
        patch.object(DatasetReader, "_build_index_mapping"),
    ):
        assert reader.try_load() is True

    assert reader.hf_dataset is sentinel


def test_try_load_rejects_a_cache_missing_some_requested_episodes():
    """Parquet loads fine but covers only part of the request: still insufficient."""
    reader = _reader(episodes=[900, 901, 999])

    with (
        patch.object(DatasetReader, "_load_hf_dataset", return_value=object()),
        patch.object(DatasetReader, "_check_cached_episodes_sufficient", return_value=False),
    ):
        assert reader.try_load() is False

    assert reader.hf_dataset is None
