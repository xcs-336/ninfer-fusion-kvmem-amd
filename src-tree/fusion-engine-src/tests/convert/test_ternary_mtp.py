from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.convert import ternary
from tools.convert.methods import cast_direct, import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import TYPE_IDS, TensorInfo
from tools.convert.sources.logical import LogicalSource


# The release's prediction block is full attention, despite text block 64's GDN
# cadence. Keep its independently specified source geometry visible in the test.
MTP_TENSORS = {
    "blk.64.attn_norm.weight": ((5120,), "F32"),
    "blk.64.post_attention_norm.weight": ((5120,), "F32"),
    "blk.64.ffn_gate.weight": ((17408, 5120), "Q8_0"),
    "blk.64.ffn_up.weight": ((17408, 5120), "Q8_0"),
    "blk.64.ffn_down.weight": ((5120, 17408), "Q8_0"),
    "blk.64.attn_q.weight": ((12288, 5120), "Q8_0"),
    "blk.64.attn_k.weight": ((1024, 5120), "Q8_0"),
    "blk.64.attn_v.weight": ((1024, 5120), "Q8_0"),
    "blk.64.attn_output.weight": ((5120, 6144), "Q8_0"),
    "blk.64.attn_q_norm.weight": ((256,), "F32"),
    "blk.64.attn_k_norm.weight": ((256,), "F32"),
    "blk.64.nextn.eh_proj.weight": ((5120, 10240), "Q8_0"),
    "blk.64.nextn.enorm.weight": ((5120,), "F32"),
    "blk.64.nextn.hnorm.weight": ((5120,), "F32"),
    "blk.64.nextn.shared_head_norm.weight": ((5120,), "F32"),
}


class _GGUF:
    """Full-size directory with only explicitly requested small rows materialised."""

    def __init__(self, mtp: bool = True):
        self.path = "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
        main = ternary.expected_tensors()
        self.kv = dict(ternary.EXPECTED_HEADER)
        self.kv["qwen35.block_count"] = 65 if mtp else 64
        self.kv["prism.hadamard.sign_widths"] = (5120, 6144, 17408)
        self.kv["prism.hadamard.sign_values"] = [1, -1] * 14336
        self.kv["prism.hadamard.weight_names"] = [
            name for name, (_, kind) in main.items()
            if kind == "PQ2_0" and name != "token_embd.weight"
        ]
        self.kv["prism.hadamard.inverse_weight_names"] = ["token_embd.weight"]
        manifest = main | (MTP_TENSORS if mtp else {})
        self.tensors = {}
        offset = 0
        for name, (shape, kind) in manifest.items():
            info = TensorInfo(name, shape, TYPE_IDS[kind], offset)
            self.tensors[name] = info
            offset += info.nbytes
        self.data_bytes_available = offset
        self.block_reads = []

    def info(self, name):
        return self.tensors[name]

    def read_direct(self, name):
        info = self.info(name)
        assert info.type_name == "F32" and name.startswith("blk.64.")
        # Exact dyadics expose the exporter's 1+w convention without rounding noise.
        return 1.0 + (np.arange(info.elements, dtype=np.float32) % 17 - 8) / 32

    def read_blocks(self, name, first, last):
        info = self.info(name)
        assert info.type_name == "Q8_0" and 0 <= first < last <= info.shape[0]
        self.block_reads.append((name, first, last))
        row = np.arange(first, last, dtype=np.uint32)[:, None, None]
        block = np.arange(info.shape[1] // 32, dtype=np.uint32)[None, :, None]
        column = np.arange(32, dtype=np.uint32)[None, None, :]
        raw = np.empty((last - first, info.shape[1] // 32, 34), dtype=np.uint8)
        raw[:, :, 2:] = (row * 17 + block * 13 + column * 7).astype(np.uint8)
        # Include signed zero, subnormal, normal and maximum finite binary16 bits.
        words = np.array(
            [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3555, 0x7BFF, 0xFBFF],
            dtype="<u2",
        )[(row[:, :, 0] + block[:, :, 0]) % 8]
        raw[:, :, :2] = words[..., None].view(np.uint8)
        return raw.reshape(last - first, -1)


def test_64_text_and_65_mtp_manifests_remain_separate_and_complete():
    main = ternary.expected_tensors()
    full = ternary.expected_tensors(mtp=True)
    assert len(main) == 851 and len(full) == 866
    assert {name: full[name] for name in full.keys() - main.keys()} == MTP_TENSORS
    assert sum(kind == "Q8_0" for _, kind in MTP_TENSORS.values()) == 8
    for mtp in (False, True):
        gguf = _GGUF(mtp)
        ternary.validate(gguf)
        assert ternary.has_mtp(gguf) == mtp
        assert gguf.block_reads == []


@pytest.mark.parametrize("name", MTP_TENSORS)
def test_every_embedded_mtp_tensor_is_required_even_for_text_only(name):
    gguf = _GGUF()
    del gguf.tensors[name]
    with pytest.raises(ValueError):
        ternary.validate(gguf)


@pytest.mark.parametrize("name", ["blk.64.unknown.weight", "blk.65.attn_norm.weight"])
def test_extra_tensors_are_rejected_instead_of_filtered(name):
    gguf = _GGUF()
    gguf.tensors[name] = TensorInfo(name, (5120,), TYPE_IDS["F32"], 0)
    with pytest.raises(ValueError, match="tensor set mismatch"):
        ternary.validate(gguf)


@pytest.mark.parametrize("mutation", ["shape", "matrix_type", "norm_type", "count", "truncated", "rotated"])
def test_known_mtp_cannot_change_geometry_format_or_rotation(mutation):
    gguf = _GGUF()
    matrix = "blk.64.nextn.eh_proj.weight"
    norm = "blk.64.nextn.enorm.weight"
    if mutation == "shape":
        gguf.tensors[matrix] = replace(gguf.tensors[matrix], shape=(10240, 5120))
    elif mutation == "matrix_type":
        gguf.tensors[matrix] = replace(gguf.tensors[matrix], type_id=TYPE_IDS["PQ2_0"])
    elif mutation == "norm_type":
        gguf.tensors[norm] = replace(gguf.tensors[norm], type_id=TYPE_IDS["BF16"])
    elif mutation == "count":
        gguf.kv["qwen35.block_count"] = 64
    elif mutation == "truncated":
        gguf.data_bytes_available -= 1
    else:
        gguf.kv["prism.hadamard.weight_names"].append(matrix)
    with pytest.raises(ValueError):
        ternary.validate(gguf)


def test_selected_mtp_rows_keep_original_scale_and_code_bytes_at_head_boundary():
    gguf = _GGUF()
    ternary.validate(gguf)
    encoded, direct = ternary.mtp_sources(gguf)
    assert len(encoded) == 9 and len(direct) == 7  # attn_q has two logical Uses.
    assert gguf.block_reads == []
    for role, indices in (
        ("query", [250, 251, 252, 253, 254, 255, 512, 513, 514, 515]),
        ("gate", [506, 507, 508, 509, 510, 511, 768, 769, 770, 771]),
    ):
        selected = encoded["mtp/layers/0/attention/" + role].read_encoded(250, 260)
        assert selected.format == "gguf_q8_0" and selected.scales.numel() == 0
        expected = np.concatenate(
            [gguf.read_blocks("blk.64.attn_q.weight", row, row + 1) for row in indices]
        )
        assert selected.codes.numpy().tobytes() == expected.tobytes()
        # The first eight scale words include all finite boundary cases verbatim.
        first_scales = selected.codes.numpy().reshape(10, -1, 34)[0, :8, :2].copy()
        assert sorted(first_scales.view("<u2").ravel().tolist()) == [
            0x0000, 0x0001, 0x03FF, 0x0400, 0x3555, 0x7BFF, 0x8000, 0xFBFF
        ]
    projection = encoded["mtp/input_projection"].read_encoded(5119, 5120)
    assert projection.codes.numpy().tobytes() == gguf.read_blocks(
        "blk.64.nextn.eh_proj.weight", 5119, 5120
    ).tobytes()
    for source in direct.values():
        expected = ((torch.arange(source.shape[0]) % 17 - 8) / 32).to(torch.bfloat16)
        assert torch.equal(source.values(), expected)


def test_embedded_mtp_uses_original_gguf_format_in_the_real_recipe():
    gguf = _GGUF()
    encoded, direct = ternary.mtp_sources(gguf)
    model = Model({"text": {"config": {}}, "mtp": {"config": {}}})
    for name, source in (encoded | direct).items():
        def forbidden_read(begin, end):
            raise AssertionError("the checkpoint values must not be requantised")

        checkpoint = LogicalSource(source.shape, "checkpoint", forbidden_read)
        model.add(Parameter(name, source.shape, checkpoint, inputs=("hidden",) if name in encoded else ()))
    recipe = Recipe(model)
    ternary._import_mtp(model, recipe, gguf)
    for name in encoded:
        selection, = recipe.selections[name]
        assert selection.format == "gguf_q8_0" and selection.method is import_encoded
        assert selection.source.label.startswith("blk.64.")
        assert recipe.policies[(name, "hidden")] == "AllowA8"
    for name in direct:
        selection, = recipe.selections[name]
        assert selection.format == "bf16" and selection.method is cast_direct
    assert recipe.auxiliary_overrides == {}  # MTP was never Hadamard-rotated.
    groups = [group for group, _ in recipe.explicit_groups]
    assert ("mtp/layers/0/attention/query", "mtp/layers/0/attention/key",
            "mtp/layers/0/attention/gate", "mtp/layers/0/attention/value") in groups
    assert ("mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up") in groups


class _LazySources(Mapping):
    def __init__(self, gguf, explicit_mtp):
        self.data = {"ternary": gguf}
        if explicit_mtp:
            self.data["mtp"] = object()

    def __getitem__(self, key):
        if key not in self.data:
            raise ValueError("a missing optional source must not be opened")
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("embedded,selected,explicit,expected", [
    (True, False, False, False),
    (True, True, False, True),
    (True, True, True, False),
    (False, True, False, False),
])
def test_component_selection_and_explicit_companion_precedence(
    monkeypatch, embedded, selected, explicit, expected
):
    gguf = _GGUF(embedded)
    model = SimpleNamespace(
        config={"hidden_size": 5120, "num_hidden_layers": 64},
        parameters={}, components={"text": {}, **({"mtp": {}} if selected else {})},
    )
    recipe = SimpleNamespace(group=lambda names: None, separate=lambda names: None)
    calls = []
    monkeypatch.setattr(ternary, "text_sources", lambda source: ({}, {}))
    monkeypatch.setattr(ternary, "_import_mtp", lambda *args: calls.append(args))
    ternary.bonsai2_27b_ternary(model, recipe, _LazySources(gguf, explicit))
    assert bool(calls) == expected
    assert gguf.block_reads == []
