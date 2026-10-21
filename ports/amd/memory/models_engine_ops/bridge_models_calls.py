"""Pure scoped transforms for the worker composer's NEW Models view. No main/I/O.

Never emits startup/program files itself. The worker composer owns final writes
and provenance; callers must fingerprint these transforms and the original TUs.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path


def _parser():
    path = Path(__file__).resolve().parent.parent / 'b1_public/prepare_public_bf16.py'
    spec = importlib.util.spec_from_file_location('models_b1_source_parser', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _include(text: str) -> str:
    marker = '#include "ninfer/ops/softmax_attention.h"'
    if text.count(marker) != 1:
        raise ValueError('expected exactly one original softmax public include')
    return text.replace(marker, marker + '\n#include "amd-models-b1/public_bf16.h"', 1)


def transform_text(text: str) -> str:
    """Both Main attn_mix arms; preserve all MTP cached/append references."""
    parser = _parser()
    marker = 'void TextContext::attn_mix('
    if text.count(marker) != 1:
        raise ValueError('Main attn_mix scope changed')
    original = parser.function(text, marker)
    replacement = parser.exact(original, 'ops::causal_softmax_attention(',
        'ops::models_engine_main_causal_softmax_attention(', 2)
    result = text.replace(original, replacement, 1)
    if result.count('ops::causal_softmax_attention(') != text.count('ops::causal_softmax_attention(') - 2:
        raise ValueError('Main-only replacement count changed')
    return _include(result)


def transform_startup(text: str) -> str:
    """Only target_body Main planning; all MTP planning remains real ordinary."""
    parser = _parser()
    marker = 'const auto target_body = [&]('
    if text.count(marker) != 1:
        raise ValueError('Main target_body workspace scope changed')
    start = text.index(marker)
    opening = text.index('{', start)
    end = parser.end_brace(text, opening)
    original = text[start:end]
    replacement = parser.exact(original,
        'ops::causal_softmax_attention_workspace_capacity_bytes(',
        '(plan.segmented_text_storage ? ops::segmented_b1_causal_softmax_attention_workspace_capacity_bytes : '
        'ops::causal_softmax_attention_workspace_capacity_bytes)(', 1)
    return _include(text[:start] + replacement + text[end:])


def transform_graph_profiles(text: str) -> str:
    """Intentionally identity: original and rebased B1 resolver bodies must match.

    The source preparer proves equality of resolver + route-label functions after
    removing only their private prefix. Shared ordinary/MTP/DFlash labels stay on
    the complete ordinary resolver; no shape-selected change to their semantics.
    """
    if text.count('return ops::causal_softmax_attention_route_family(') != 1:
        raise ValueError('original Graph route-family query changed')
    return text
