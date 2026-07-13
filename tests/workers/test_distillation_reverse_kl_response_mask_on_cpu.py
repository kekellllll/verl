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
"""Regression guard for the torch 2.11 nested-tensor ``response_mask``
alignment fix in ``compute_distillation_loss_reverse_kl_estimator``
(``verl/trainer/distillation/losses.py``).

Bug being locked down (observed on Qwen3.5-35B-A3B OPD smoke, 2026-07-12):
  - ``student_log_probs`` / ``teacher_log_probs`` are produced by
    ``no_padding_2_padding``, which for nested prompts/responses pads
    the response axis to ``max_response_len`` = ``max(response_lens)``
    from ``response_ids.offsets().diff()``.
  - ``response_mask`` (when nested) is padded via
    ``to_padded_tensor(False)``, which pads only to that nested tensor's
    own max ragged length.
  - Under torch 2.11 the two baselines can disagree for some batches;
    the pre-patch code raised a bare ``assert teacher.shape ==
    student.shape == mask.shape`` and crashed training in step 2 while
    step 1 happened to be shape-consistent.

The patch (around line 389 of ``losses.py``) aligns ``response_mask_bool``
to the student axis before the assert: pad with ``False`` when shorter,
truncate when longer. Padded positions are ``mask=False`` and by
construction contribute nothing to any masked reduction
(``losses[response_mask_bool].abs().mean()``).

This file locks down three contracts:
  1. ``shape_consistent_is_noop``: when the nested response_mask already
     pads to the same length as ``no_padding_2_padding``, the alignment
     block is a pure no-op and the loss returns a finite tensor of
     shape ``(bsz, max_response_len)``.
  2. ``pad_shorter_masked_metric_stable``: when we force the nested mask
     to be strictly shorter than the student axis, the patch pads with
     ``False``; the masked metric ``distillation/abs_loss`` is
     numerically identical to a shape-consistent baseline computed on
     the same underlying data (padded positions do not contribute).
  3. ``truncate_longer_masked_metric_stable``: symmetric case — when
     the nested mask is longer than the student axis, the patch
     truncates; padded positions on the trailing edge (mask=False)
     don't contribute to the masked metric.

CPU only, no GPU kernels involved. Runs in a few seconds.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import (
    compute_distillation_loss_reverse_kl_estimator,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------
def _make_distillation_config() -> DistillationConfig:
    """Minimal DistillationConfig with k1 (reverse-KL) loss_mode."""
    loss_cfg = DistillationLossConfig(
        loss_mode="k1",
        topk=64,
        use_task_rewards=False,
        use_policy_gradient=True,
        loss_max_clamp=10.0,
        log_prob_min_clamp=-10.0,
    )
    return DistillationConfig(distillation_loss=loss_cfg)


def _build_inputs(prompt_lens, response_lens, mask_len_override=None):
    """Construct real packed inputs for
    ``compute_distillation_loss_reverse_kl_estimator``.

    Uses nested prompts/responses (matches production OPD data path); this
    means ``no_padding_2_padding`` reads lengths from
    ``response_ids.offsets().diff()`` and derives
    ``max_response_len = max(response_lens)``.

    Args:
        prompt_lens: list[int], per-sample prompt token counts.
        response_lens: list[int], per-sample response token counts.
        mask_len_override: if set, override the length of each
            per-sample mask segment used to construct the nested
            response_mask, decoupling its ``to_padded_tensor`` baseline
            from ``max(response_lens)``. This is how we hit the pad /
            truncate branches deterministically.

    Returns:
        (model_output, data) accepted by the loss function under test.
    """
    bsz = len(prompt_lens)
    assert len(response_lens) == bsz
    total_nnz = sum(p + r for p, r in zip(prompt_lens, response_lens))

    # Deterministic packed logprobs across the whole batch.
    student_packed = torch.arange(total_nnz, dtype=torch.float32) * 0.01 + 0.3
    # teacher_logprobs is squeezed via .squeeze(-1) inside the loss, so
    # last dim must be present (see losses.py line ~386).
    teacher_packed = (
        torch.arange(total_nnz, dtype=torch.float32) * 0.007 - 0.2
    ).reshape(total_nnz, 1)

    # Nested prompts / responses (real OPD data layout). Their concrete
    # token values don't matter for this loss codepath — only lengths do.
    prompt_list = [torch.zeros(p, dtype=torch.long) for p in prompt_lens]
    response_list = [torch.zeros(r, dtype=torch.long) for r in response_lens]
    prompts_nested = torch.nested.as_nested_tensor(
        prompt_list, layout=torch.jagged
    )
    responses_nested = torch.nested.as_nested_tensor(
        response_list, layout=torch.jagged
    )

    # Response mask construction — the invariant that makes the three
    # tests comparable: the set of ``mask=True`` positions per sample is
    # FIXED to the first ``r_i - 2`` positions (with ``r_i =
    # response_lens[i]``) regardless of ``mask_len_override``. Only the
    # trailing ``False`` region grows / shrinks, so any ``abs_loss``
    # discrepancy between baseline and pad/truncate paths would be caused
    # solely by the alignment block touching mask=True positions — which
    # the patch must NOT do.
    #
    # ``mask_len_override`` sets the per-sample segment length used to
    # build the nested tensor; when None, each segment length = r_i
    # (so nested-baseline == max(response_lens), the no-op path).
    seg_lens = (
        response_lens
        if mask_len_override is None
        else [mask_len_override] * bsz
    )
    n_true_per_sample = [max(1, r - 2) for r in response_lens]
    mask_segs = []
    for i, L in enumerate(seg_lens):
        seg = torch.zeros(L)
        # Cap True count by segment length so we never over-write; this
        # is only tight for ``truncate_longer`` (L > r), where any
        # extra positions are False by construction.
        n_true = min(n_true_per_sample[i], L)
        seg[:n_true] = 1.0
        mask_segs.append(seg)
    rmask_nested = torch.nested.nested_tensor(
        mask_segs, layout=torch.jagged
    )

    data = TensorDict(
        {
            "prompts": prompts_nested,
            "responses": responses_nested,
            "teacher_logprobs": teacher_packed,
            "response_mask": rmask_nested,
        },
        batch_size=[],
    )
    model_output = {"log_probs": student_packed}
    return model_output, data


def _run(model_output, data, cfg):
    losses, metrics = compute_distillation_loss_reverse_kl_estimator(
        None, cfg, model_output, data
    )
    # ``metrics["distillation/abs_loss"]`` is a ``verl.utils.metric.utils.Metric``
    # (MEAN aggregator). ``aggregate()`` returns the reduced Python float.
    return (
        losses.detach().clone(),
        float(metrics["distillation/abs_loss"].aggregate()),
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_reverse_kl_shape_consistent_is_noop():
    """When nested response_mask's ``to_padded_tensor`` baseline equals
    ``max(response_lens)``, ``_cur == _tgt`` and the alignment block is
    a pure no-op. The loss must return a finite tensor of shape
    ``(bsz, max(response_lens))``.
    """
    cfg = _make_distillation_config()
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    bsz = len(prompt_lens)
    max_resp = max(response_lens)  # == 7

    mo, data = _build_inputs(prompt_lens, response_lens)
    losses, abs_loss = _run(mo, data, cfg)

    assert losses.shape == (bsz, max_resp), (
        f"unexpected shape {losses.shape}, expected ({bsz},{max_resp})"
    )
    assert torch.isfinite(losses).all(), "losses has NaN/Inf"
    assert torch.isfinite(torch.tensor(abs_loss)), f"abs_loss not finite: {abs_loss}"


def test_reverse_kl_pad_shorter_matches_baseline():
    """Force the nested response_mask baseline to be strictly SHORTER than
    ``max(response_lens)`` by setting every mask segment length to
    ``max(response_lens) - 2``. The patch must pad with ``False`` so the
    shape check passes, and — because mask=False positions don't
    contribute — the masked ``abs_loss`` must equal the shape-consistent
    baseline.
    """
    cfg = _make_distillation_config()
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    bsz = len(prompt_lens)
    max_resp = max(response_lens)  # 7

    # Baseline: shape-consistent
    mo_base, data_base = _build_inputs(prompt_lens, response_lens)
    _, abs_base = _run(mo_base, data_base, cfg)

    # Triggered: mask baseline = max_resp - 2 = 5 < 7 → patch pads with False
    mo_pad, data_pad = _build_inputs(
        prompt_lens, response_lens, mask_len_override=max_resp - 2
    )
    losses_pad, abs_pad = _run(mo_pad, data_pad, cfg)

    assert losses_pad.shape == (bsz, max_resp), (
        f"unexpected shape after pad: {losses_pad.shape}"
    )
    assert torch.isfinite(losses_pad).all()
    # Padded positions carry mask=False by patch construction; they must
    # not shift the masked reduction.
    assert abs_pad == pytest.approx(abs_base, rel=0.0, abs=1e-10), (
        f"masked abs_loss changed with False-padding: "
        f"base={abs_base}  pad={abs_pad}"
    )


def test_reverse_kl_truncate_longer_masked_metric_stable():
    """Symmetric to the pad case: force nested response_mask baseline to
    be strictly LONGER than ``max(response_lens)`` (mask carries trailing
    False entries anyway by construction). The patch must truncate to
    the student axis, the shape check must pass, and the masked
    ``abs_loss`` must equal the shape-consistent baseline (the truncated
    positions are all mask=False in this construction).
    """
    cfg = _make_distillation_config()
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    bsz = len(prompt_lens)
    max_resp = max(response_lens)  # 7

    mo_base, data_base = _build_inputs(prompt_lens, response_lens)
    _, abs_base = _run(mo_base, data_base, cfg)

    # Triggered: mask baseline = max_resp + 3 = 10 > 7 → patch truncates
    mo_tr, data_tr = _build_inputs(
        prompt_lens, response_lens, mask_len_override=max_resp + 3
    )
    losses_tr, abs_tr = _run(mo_tr, data_tr, cfg)

    assert losses_tr.shape == (bsz, max_resp), (
        f"unexpected shape after truncate: {losses_tr.shape}"
    )
    assert torch.isfinite(losses_tr).all()
    # Truncated positions in this construction are all mask=False.
    assert abs_tr == pytest.approx(abs_base, rel=0.0, abs=1e-10), (
        f"masked abs_loss changed with truncation: "
        f"base={abs_base}  truncate={abs_tr}"
    )
