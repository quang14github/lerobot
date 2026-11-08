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
"""Stop requests, setting fingerprints and resumable CSV logs of federated runs."""

import os
import signal

import pytest

from lerobot.federated.run_state import CsvLog, StopRequest, compare_settings, config_fingerprint


def test_ctrl_c_requests_a_clean_stop_and_a_second_one_aborts():
    stop = StopRequest()
    try:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.requested and stop.signal_name == "SIGINT"
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    finally:
        stop.restore()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_sigterm_requests_a_clean_stop():
    stop = StopRequest()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)  # repeated: still just a request
        assert stop.requested
    finally:
        stop.restore()


def test_fingerprint_ignores_benign_settings_only():
    base = {"fl": {"rounds": 200, "local_steps": 250, "auto_resume": True}, "num_workers": 3, "seed": 1000}
    benign = {**base, "num_workers": 8, "fl": {**base["fl"], "auto_resume": False}}
    changed = {**base, "fl": {**base["fl"], "local_steps": 100}}
    fp, settings = config_fingerprint(base)
    assert config_fingerprint(benign)[0] == fp
    assert config_fingerprint(changed)[0] != fp
    assert compare_settings(settings, config_fingerprint(changed)[1]) == ["fl.local_steps: 250 -> 100"]


def test_csv_log_drops_rows_of_rounds_a_resume_replays(tmp_path):
    log = CsvLog(tmp_path / "rounds.csv", ["round", "loss"])
    log.append([{"round": r, "loss": r / 10} for r in (1, 2, 3)])
    log.truncate_after(2)
    log.append([{"round": 3, "loss": 0.9}])
    assert (tmp_path / "rounds.csv").read_text().splitlines() == ["round,loss", "1,0.1", "2,0.2", "3,0.9"]
