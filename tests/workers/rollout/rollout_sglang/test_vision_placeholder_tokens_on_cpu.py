# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""SGLang must not sample vision placeholders without backing multimodal data."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.rollout.sglang_rollout.utils import (
    add_vision_placeholder_sampling_controls,
    get_vision_placeholder_sampling_controls,
)


def test_text_only_processor_does_not_require_sglang_custom_processor():
    assert get_vision_placeholder_sampling_controls(None) == (None, [])


def test_request_controls_merge_existing_disallowed_tokens_without_mutation():
    request = {
        "sampling_params": {
            "temperature": 0.8,
            "custom_params": {"token_ids": [7], "trace": "keep"},
        }
    }

    updated = add_vision_placeholder_sampling_controls(request, "serialized", [2, 4, 7])

    assert updated["custom_logit_processor"] == "serialized"
    assert updated["sampling_params"]["custom_params"] == {
        "token_ids": [7, 2, 4],
        "trace": "keep",
    }
    assert request["sampling_params"]["custom_params"]["token_ids"] == [7]


def test_request_without_vision_tokens_is_unchanged():
    request = {"sampling_params": {"temperature": 0.8}}

    assert add_vision_placeholder_sampling_controls(request, None, []) is request


def test_vision_processor_serializes_sglang_disallowed_tokens():
    pytest.importorskip("sglang")
    from sglang.srt.sampling.custom_logit_processor import (
        CustomLogitProcessor,
        DisallowedTokensLogitsProcessor,
    )

    processor = SimpleNamespace(image_token_id=151655, video_token_id=151656)
    serialized, token_ids = get_vision_placeholder_sampling_controls(processor)

    assert token_ids == [151655, 151656]
    assert isinstance(serialized, str)
    assert isinstance(CustomLogitProcessor.from_str(serialized), DisallowedTokensLogitsProcessor)


def test_disallowed_tokens_processor_masks_only_requested_ids():
    pytest.importorskip("sglang")
    from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor

    logits = torch.arange(8, dtype=torch.float32).repeat(2, 1)
    baseline = logits.clone()
    params = [{"token_ids": [2, 4]}, {"token_ids": [2, 4]}]

    output = DisallowedTokensLogitsProcessor()(logits, params)

    assert torch.isneginf(output[:, [2, 4]]).all()
    torch.testing.assert_close(output[:, [0, 1, 3, 5, 6, 7]], baseline[:, [0, 1, 3, 5, 6, 7]])
