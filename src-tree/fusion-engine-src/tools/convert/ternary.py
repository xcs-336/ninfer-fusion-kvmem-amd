"""Ternary Bonsai 2 27B: a PrismML PQ2_0 GGUF as the text tower of a Qwen3.8-27B artifact.

The GGUF stores every text projection except the GDN A/B controls as Hadamard-rotated ternary
rows with one binary16 scale per 128 columns, the token-embedding table rotated as well, and
llama.cpp's exporter conventions: GDN value heads in tiled order, zero-centred norms as `1 + w`
and `ssm_a = -exp(A_log)`. This module restores the grouped value-head order and the primal
norm and A_log values, and exposes the rotated projections as encoded T2 rows whose Uses carry
the sign vector of their input width, so the recipe stores them without rounding. The token table
is stored the same way, inverse-rotated; the runtime restores each gathered row with the
hidden-width signs. An optional, complete blk.64 MTP head is recognised only in its registered
Q8_0/F32 geometry. Selecting the MTP component imports those matrix rows as the original GGUF
blocks, without the text tower's Hadamard signs or another quantisation. A text-only selection
still validates the complete source manifest. Vision, frontend resources and the DFlash2 adapter
come from the companions given to `--model` and `--source dflash2`; an explicit `--source mtp`
takes precedence over the embedded head. A GGUF without MTP retains the checkpoint MTP source.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .methods import AuxiliaryValue, cast_direct, grouped_absmax, import_encoded
from .official_recipes import Q4, _optional
from .sources.gguf import GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source

T2 = "t2_g128_fp16"

# The DFlash2 adapter projections a ternary artifact encodes as Q4. The fused query/key/value
# projection, which the context key/value Uses share, stays Q8 for its Q8-only three-output op.
DFLASH2_Q4_PROJECTIONS = (
    "/feature_projection",
    "/attention/output",
    "/mlp/gate",
    "/mlp/up",
    "/mlp/down",
)

HIDDEN = 5120
INTERMEDIATE = 17408
VOCABULARY = 248320
LAYERS = 64
ATTENTION_HEADS = 24
ATTENTION_HEAD_DIM = 256
ATTENTION_QUERY_ROWS = ATTENTION_HEADS * ATTENTION_HEAD_DIM
ATTENTION_KV_ROWS = 1024
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_KEY_DIM = GDN_KEY_HEADS * GDN_HEAD_DIM
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_DIM
GDN_CHANNELS = 2 * GDN_KEY_DIM + GDN_VALUE_DIM
GDN_TAPS = 4
HADAMARD_BLOCK = 1024
SIGN_WIDTHS = (HIDDEN, ATTENTION_QUERY_ROWS, INTERMEDIATE)

EXPECTED_HEADER = {
    "general.architecture": "qwen35",
    "qwen35.block_count": LAYERS,
    "qwen35.embedding_length": HIDDEN,
    "qwen35.feed_forward_length": INTERMEDIATE,
    "qwen35.attention.head_count": ATTENTION_HEADS,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": ATTENTION_HEAD_DIM,
    "qwen35.attention.value_length": ATTENTION_HEAD_DIM,
    "qwen35.ssm.conv_kernel": GDN_TAPS,
    "qwen35.ssm.state_size": GDN_HEAD_DIM,
    "qwen35.ssm.group_count": GDN_KEY_HEADS,
    "qwen35.ssm.time_step_rank": GDN_VALUE_HEADS,
    "qwen35.ssm.inner_size": GDN_VALUE_DIM,
    "qwen35.full_attention_interval": 4,
    "prism.hadamard.version": 1,
    "prism.hadamard.block_size": HADAMARD_BLOCK,
    "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
    "prism.hadamard.axis": "input-last-dimension",
    "prism.hadamard.sign_mode": "explicit",
    "prism.hadamard.gdn_v_grouped": True,
}


def full_attention(layer: int) -> bool:
    return layer % 4 == 3


def expected_tensors(mtp: bool = False) -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape and GGUF type of every source tensor."""

    out = {
        "token_embd.weight": ((VOCABULARY, HIDDEN), "PQ2_0"),
        "output.weight": ((VOCABULARY, HIDDEN), "PQ2_0"),
        "output_norm.weight": ((HIDDEN,), "F32"),
    }
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        out[p + "attn_norm.weight"] = ((HIDDEN,), "F32")
        out[p + "post_attention_norm.weight"] = ((HIDDEN,), "F32")
        out[p + "ffn_gate.weight"] = ((INTERMEDIATE, HIDDEN), "PQ2_0")
        out[p + "ffn_up.weight"] = ((INTERMEDIATE, HIDDEN), "PQ2_0")
        out[p + "ffn_down.weight"] = ((HIDDEN, INTERMEDIATE), "PQ2_0")
        if full_attention(layer):
            out[p + "attn_q.weight"] = ((2 * ATTENTION_QUERY_ROWS, HIDDEN), "PQ2_0")
            out[p + "attn_k.weight"] = ((ATTENTION_KV_ROWS, HIDDEN), "PQ2_0")
            out[p + "attn_v.weight"] = ((ATTENTION_KV_ROWS, HIDDEN), "PQ2_0")
            out[p + "attn_output.weight"] = ((HIDDEN, ATTENTION_QUERY_ROWS), "PQ2_0")
            out[p + "attn_q_norm.weight"] = ((ATTENTION_HEAD_DIM,), "F32")
            out[p + "attn_k_norm.weight"] = ((ATTENTION_HEAD_DIM,), "F32")
        else:
            out[p + "attn_qkv.weight"] = ((GDN_CHANNELS, HIDDEN), "PQ2_0")
            out[p + "attn_gate.weight"] = ((GDN_VALUE_DIM, HIDDEN), "PQ2_0")
            out[p + "ssm_out.weight"] = ((HIDDEN, GDN_VALUE_DIM), "PQ2_0")
            out[p + "ssm_alpha.weight"] = ((GDN_VALUE_HEADS, HIDDEN), "BF16")
            out[p + "ssm_beta.weight"] = ((GDN_VALUE_HEADS, HIDDEN), "BF16")
            out[p + "ssm_a"] = ((GDN_VALUE_HEADS,), "F32")
            out[p + "ssm_dt.bias"] = ((GDN_VALUE_HEADS,), "F32")
            out[p + "ssm_conv1d.weight"] = ((GDN_CHANNELS, GDN_TAPS), "F32")
            out[p + "ssm_norm.weight"] = ((GDN_HEAD_DIM,), "F32")
    if mtp:
        # This release adds a full-attention prediction block after the 64 text blocks.
        # Its unrotated Q8_0 matrices must not enter the PQ2_0 Hadamard tensor set.
        p = f"blk.{LAYERS}."
        out |= {
            p + "attn_norm.weight": ((HIDDEN,), "F32"),
            p + "post_attention_norm.weight": ((HIDDEN,), "F32"),
            p + "ffn_gate.weight": ((INTERMEDIATE, HIDDEN), "Q8_0"),
            p + "ffn_up.weight": ((INTERMEDIATE, HIDDEN), "Q8_0"),
            p + "ffn_down.weight": ((HIDDEN, INTERMEDIATE), "Q8_0"),
            p + "attn_q.weight": ((2 * ATTENTION_QUERY_ROWS, HIDDEN), "Q8_0"),
            p + "attn_k.weight": ((ATTENTION_KV_ROWS, HIDDEN), "Q8_0"),
            p + "attn_v.weight": ((ATTENTION_KV_ROWS, HIDDEN), "Q8_0"),
            p + "attn_output.weight": ((HIDDEN, ATTENTION_QUERY_ROWS), "Q8_0"),
            p + "attn_q_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
            p + "attn_k_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
            p + "nextn.eh_proj.weight": ((HIDDEN, 2 * HIDDEN), "Q8_0"),
            p + "nextn.enorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.hnorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.shared_head_norm.weight": ((HIDDEN,), "F32"),
        }
    return out


def has_mtp(gguf: GGUFFile) -> bool:
    """Whether the source declares the registered embedded prediction head.

    This marker does not relax validation: block count, every MTP tensor and every
    other source tensor are checked by :func:`validate`, even for text-only imports.
    """

    return f"blk.{LAYERS}.nextn.eh_proj.weight" in gguf.tensors


def validate(gguf: GGUFFile) -> None:
    """Refuse any GGUF that is not the registered ternary Bonsai 2 27B layout."""

    mtp = has_mtp(gguf)
    for key, expected in EXPECTED_HEADER.items():
        if key == "qwen35.block_count":
            expected = LAYERS + int(mtp)
        if gguf.kv.get(key) != expected:
            raise ValueError(
                f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}"
            )
    widths = tuple(
        int(width) for width in gguf.kv.get("prism.hadamard.sign_widths", ())
    )
    if widths != SIGN_WIDTHS:
        raise ValueError(f"{gguf.path}: sign widths {widths}, expected {SIGN_WIDTHS}")
    values = np.asarray(gguf.kv.get("prism.hadamard.sign_values", ()), dtype=np.int64)
    if values.size != sum(SIGN_WIDTHS) or not np.all(np.abs(values) == 1):
        raise ValueError(
            f"{gguf.path}: sign values must be {sum(SIGN_WIDTHS)} entries of +-1"
        )
    expected = expected_tensors(mtp)
    rotated = {name for name, (_, kind) in expected.items() if kind == "PQ2_0"} - {
        "token_embd.weight"
    }
    if set(gguf.kv.get("prism.hadamard.weight_names", ())) != rotated:
        raise ValueError(
            f"{gguf.path}: the rotated tensor set differs from the registered one"
        )
    if tuple(gguf.kv.get("prism.hadamard.inverse_weight_names", ())) != (
        "token_embd.weight",
    ):
        raise ValueError(
            f"{gguf.path}: only token_embd.weight may be stored inverse-rotated"
        )
    if set(gguf.tensors) != set(expected):
        missing = sorted(set(expected) - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )
    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        if info.shape != shape or info.type_name != kind:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def sign_vectors(gguf: GGUFFile) -> dict[int, torch.Tensor]:
    """The +-1 sign vector of every rotated input width."""

    values = np.asarray(gguf.kv["prism.hadamard.sign_values"], dtype=np.float32)
    out, offset = {}, 0
    for width in SIGN_WIDTHS:
        out[width] = torch.from_numpy(
            np.ascontiguousarray(values[offset : offset + width])
        )
        offset += width
    return out


def sign_auxiliary(signs: torch.Tensor) -> AuxiliaryValue:
    words = signs.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
    return AuxiliaryValue("bf16", (int(signs.numel()),), words.astype("<i2").tobytes())


def tiled_to_grouped_permutation() -> np.ndarray:
    """For grouped value head `h = k * 3 + r`, the exporter's tiled position `r * 16 + k`.

    llama.cpp's exporter stores value heads tiled so that a plain repeat broadcasts the key
    heads; NInfer pairs value head `h` with key head `h // 3` and wants the grouped order.
    """

    heads = np.arange(GDN_VALUE_HEADS)
    per_key = GDN_VALUE_HEADS // GDN_KEY_HEADS
    return (heads % per_key) * GDN_KEY_HEADS + heads // per_key


def untile(array: np.ndarray, head_dim: int) -> np.ndarray:
    """Reorder axis 0 from the tiled value-head order back to grouped."""

    if array.shape[0] != GDN_VALUE_HEADS * head_dim:
        raise ValueError(
            f"axis 0 has {array.shape[0]} entries, expected {GDN_VALUE_HEADS * head_dim}"
        )
    view = array.reshape(GDN_VALUE_HEADS, head_dim, *array.shape[1:])
    return np.ascontiguousarray(
        view[tiled_to_grouped_permutation()].reshape(array.shape)
    )


RowMap = Callable[[int, int], np.ndarray]


def rows(base: int = 0) -> RowMap:
    return lambda begin, end: np.arange(begin, end, dtype=np.int64) + base


def attention_rows(gate: bool) -> RowMap:
    """The query (or output gate) rows of the head-interleaved `attn_q`."""

    d = ATTENTION_HEAD_DIM

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return (r // d) * 2 * d + (d if gate else 0) + r % d

    return select


def untiled_rows(base: int) -> RowMap:
    """Grouped value-head rows of a tiled family starting at source row `base`."""

    permutation = tiled_to_grouped_permutation()

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return base + permutation[r // GDN_HEAD_DIM] * GDN_HEAD_DIM + r % GDN_HEAD_DIM

    return select


def _flat(read_rows, k: int):
    def read(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        values = read_rows(first, last).reshape(-1)
        return values[begin - first * k : end - first * k]

    return read


def ternary_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Rotated ternary rows as encoded T2 words, and their exact values for any other method."""

    k = shape[1]

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_ternary(tensor, low, high)
        return EncodedRows(
            T2,
            torch.from_numpy(np.ascontiguousarray(blocks.values[index - low])),
            torch.from_numpy(np.ascontiguousarray(blocks.scales[index - low])),
        )

    def values(first: int, last: int) -> torch.Tensor:
        words = encoded(first, last)
        return words.codes.float() * words.scales.float().unsqueeze(-1)

    return LogicalSource(shape, f"{tensor}{list(shape)}", _flat(values, k), encoded)


def _direct(values: np.ndarray, dtype: torch.dtype, label: str) -> LogicalSource:
    return array_source(torch.from_numpy(np.ascontiguousarray(values)).to(dtype), label)


def _norm(gguf: GGUFFile, tensor: str, offset: bool) -> LogicalSource:
    words = gguf.read_direct(tensor).astype(np.float32)
    return _direct(words - 1.0 if offset else words, torch.bfloat16, tensor)


def text_sources(
    gguf: GGUFFile,
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded T2 sources and direct sources of every GGUF-backed text parameter."""

    encoded = {
        "text/output_head": ternary_source(
            gguf, "output.weight", (VOCABULARY, HIDDEN), rows()
        ),
        # Stored inverse-rotated: the runtime restores each gathered row as s * FWHT(z).
        "text/token_embedding": ternary_source(
            gguf, "token_embd.weight", (VOCABULARY, HIDDEN), rows()
        ),
    }
    direct = {"text/final_norm": _norm(gguf, "output_norm.weight", True)}
    for layer in range(LAYERS):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        direct[p + "input_norm"] = _norm(gguf, g + "attn_norm.weight", True)
        direct[p + "post_attention_norm"] = _norm(
            gguf, g + "post_attention_norm.weight", True
        )
        encoded[p + "mlp/gate"] = ternary_source(
            gguf, g + "ffn_gate.weight", (INTERMEDIATE, HIDDEN), rows()
        )
        encoded[p + "mlp/up"] = ternary_source(
            gguf, g + "ffn_up.weight", (INTERMEDIATE, HIDDEN), rows()
        )
        encoded[p + "mlp/down"] = ternary_source(
            gguf, g + "ffn_down.weight", (HIDDEN, INTERMEDIATE), rows()
        )
        if full_attention(layer):
            a = p + "attention/"
            q_shape, kv_shape = (
                (ATTENTION_QUERY_ROWS, HIDDEN),
                (ATTENTION_KV_ROWS, HIDDEN),
            )
            encoded[a + "query"] = ternary_source(
                gguf, g + "attn_q.weight", q_shape, attention_rows(False)
            )
            encoded[a + "gate"] = ternary_source(
                gguf, g + "attn_q.weight", q_shape, attention_rows(True)
            )
            encoded[a + "key"] = ternary_source(
                gguf, g + "attn_k.weight", kv_shape, rows()
            )
            encoded[a + "value"] = ternary_source(
                gguf, g + "attn_v.weight", kv_shape, rows()
            )
            encoded[a + "output"] = ternary_source(
                gguf, g + "attn_output.weight", (HIDDEN, ATTENTION_QUERY_ROWS), rows()
            )
            direct[a + "query_norm"] = _norm(gguf, g + "attn_q_norm.weight", True)
            direct[a + "key_norm"] = _norm(gguf, g + "attn_k_norm.weight", True)
            continue
        d = p + "gdn/"
        qkv = g + "attn_qkv.weight"
        encoded[d + "query"] = ternary_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows())
        encoded[d + "key"] = ternary_source(
            gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows(GDN_KEY_DIM)
        )
        encoded[d + "value"] = ternary_source(
            gguf, qkv, (GDN_VALUE_DIM, HIDDEN), untiled_rows(2 * GDN_KEY_DIM)
        )
        encoded[d + "z"] = ternary_source(
            gguf, g + "attn_gate.weight", (GDN_VALUE_DIM, HIDDEN), untiled_rows(0)
        )
        # ssm_out keeps its grouped column order in the GGUF (prism.hadamard.gdn_v_grouped).
        encoded[d + "output"] = ternary_source(
            gguf, g + "ssm_out.weight", (HIDDEN, GDN_VALUE_DIM), rows()
        )
        for role, tensor in (
            ("a_projection", "ssm_alpha.weight"),
            ("b_projection", "ssm_beta.weight"),
        ):
            words = untile(gguf.read_bf16_words(g + tensor), 1)
            direct[d + role] = array_source(
                torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(
                    torch.bfloat16
                ),
                g + tensor,
            )
        ssm_a = untile(gguf.read_direct(g + "ssm_a"), 1).astype(np.float64)
        if not np.all(ssm_a < 0):
            raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
        direct[d + "a_log"] = _direct(
            np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a"
        )
        direct[d + "dt_bias"] = _direct(
            untile(gguf.read_direct(g + "ssm_dt.bias"), 1),
            torch.float32,
            g + "ssm_dt.bias",
        )
        taps = gguf.read_direct(g + "ssm_conv1d.weight")
        channels = np.concatenate(
            [taps[: 2 * GDN_KEY_DIM], untile(taps[2 * GDN_KEY_DIM :], GDN_HEAD_DIM)]
        )
        direct[d + "convolution"] = _direct(
            channels.T, torch.bfloat16, g + "ssm_conv1d.weight"
        )
        direct[d + "norm"] = _norm(gguf, g + "ssm_norm.weight", False)
    return encoded, direct


def mtp_sources(
    gguf: GGUFFile,
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """The selected embedded MTP head, preserving each Q8_0 block's scale and codes.

    The GGUF block importer already owns the exporter's full-attention row mapping
    and norm convention. Import locally because that module uses the text geometry
    and row maps defined here. The caller validates the full GGUF before selection.
    """

    from .gguf_blocks import _attention, _layer, block_source

    g, p = f"blk.{LAYERS}.", "mtp/layers/0/"
    encoded, direct = _layer(gguf, g, p)
    attention, norms = _attention(gguf, g, p + "attention/")
    encoded |= attention
    direct |= norms
    encoded["mtp/input_projection"] = block_source(
        gguf, g + "nextn.eh_proj.weight", (HIDDEN, 2 * HIDDEN), rows()
    )
    for role, tensor in (
        ("embedding_norm", "nextn.enorm.weight"),
        ("hidden_norm", "nextn.hnorm.weight"),
        ("final_norm", "nextn.shared_head_norm.weight"),
    ):
        direct["mtp/" + role] = _norm(gguf, g + tensor, True)
    return encoded, direct


def _import_mtp(model, recipe, gguf: GGUFFile) -> None:
    encoded, direct = mtp_sources(gguf)
    for name, source in encoded.items():
        recipe.assign(
            name,
            format=source.read_encoded(0, 1).format,
            method=import_encoded,
            source=source,
            activation_policy="AllowA8",
        )
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    # All eight source matrices are Q8_0 under this recipe's validated contract.
    a = "mtp/layers/0/attention/"
    recipe.group([a + role for role in ("query", "key", "gate", "value")])
    p = "mtp/layers/0/mlp/"
    recipe.group([p + "gate", p + "up"])


def bonsai2_27b_ternary(model, recipe, sources):
    """Ternary Bonsai 2 27B from `--source ternary=Ternary-Bonsai-2-27B-PQ2_0.gguf`."""

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != HIDDEN
        or config["num_hidden_layers"] != LAYERS
    ):
        raise ValueError(
            "the ternary Bonsai 2 recipe requires the Qwen3.8-27B Dense geometry"
        )
    gguf = sources["ternary"]
    validate(gguf)
    signs = sign_vectors(gguf)
    _optional(model, recipe)
    # Against a 2.125-bit target, a Q8 drafter is a large share of every draft step's bytes, so the
    # DFlash2 adapter's feature, output and MLP projections take Q4 (its dynamic-convolution kernel
    # projections and the candidate selector keep the formats _optional gives them).
    for name in model.parameters:
        if name.startswith("dflash2/") and name.endswith(DFLASH2_Q4_PROJECTIONS):
            recipe.assign(name, format=Q4, method=grouped_absmax)
    encoded, direct = text_sources(gguf)
    for name, source in encoded.items():
        recipe.assign(name, format=T2, method=import_encoded, source=source)
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    # The runtime's T2 input projections take the Q/K + gate/V and q/k + v/z parent pairs.
    for layer in range(LAYERS):
        p = f"text/layers/{layer}/"
        if full_attention(layer):
            recipe.group([p + "attention/query", p + "attention/key"])
            recipe.group([p + "attention/gate", p + "attention/value"])
        else:
            recipe.group([p + "gdn/query", p + "gdn/key"])
            recipe.group([p + "gdn/value", p + "gdn/z"])
            recipe.separate([p + "gdn/a_projection", p + "gdn/b_projection"])
        recipe.group([p + "mlp/gate", p + "mlp/up"])
    auxiliaries = {width: sign_auxiliary(values) for width, values in signs.items()}
    for name in encoded:
        parameter = model.parameters[name]
        for input_name in parameter.inputs:
            recipe.use(
                name,
                input_name,
                auxiliaries={"hadamard_signs": auxiliaries[parameter.shape[1]]},
            )
    # SourceInputs is a lazy Mapping whose __getitem__ reports missing sources with
    # ValueError. Iterate its keys instead of triggering a missing-source lookup.
    if "mtp" in model.components and has_mtp(gguf) and "mtp" not in set(sources):
        _import_mtp(model, recipe, gguf)


RECIPES = {"bonsai2_27b_ternary": bonsai2_27b_ternary}

__all__ = [
    "EXPECTED_HEADER",
    "RECIPES",
    "attention_rows",
    "bonsai2_27b_ternary",
    "expected_tensors",
    "has_mtp",
    "mtp_sources",
    "sign_vectors",
    "ternary_source",
    "text_sources",
    "tiled_to_grouped_permutation",
    "untile",
    "untiled_rows",
    "validate",
]
