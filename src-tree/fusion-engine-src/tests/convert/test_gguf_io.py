from __future__ import annotations

import os

import numpy as np
import pytest

from tools.convert.sources.gguf import GGUFFile, TYPE_F32, write_gguf


def test_exact_disjoint_ranges_and_views_survive_close(tmp_path):
    words = np.array([0, 0x80000000, 0x3F800000, 0x007FFFFF, 0x7FC00001, 0xFF800000],
                     dtype='<u4')
    path = tmp_path / 'ranges.gguf'
    write_gguf(path, {}, [('values', (2, 3), TYPE_F32, words.tobytes())])
    with GGUFFile(path) as source:
        tail = source.tensor_bytes('values', 16, 24)
        first = source.tensor_bytes('values', 0, 8)
        crossing = source.tensor_bytes('values', 4, 20)
        assert not first.flags.writeable
        assert source.tensor_bytes('values', 8, 8).size == 0
    source.close()  # Cleanup remains idempotent.
    assert tail.tobytes() == words[4:].tobytes()
    assert first.tobytes() == words[:2].tobytes()
    assert crossing.tobytes() == words[1:5].tobytes()
    with pytest.raises(ValueError, match='closed'):
        source.tensor_bytes('values')


def test_short_read_after_source_truncation_is_reported(tmp_path):
    path = tmp_path / 'short.gguf'
    values = np.arange(12, dtype='<f4')
    write_gguf(path, {}, [('values', (3, 4), TYPE_F32, values.tobytes())])
    with GGUFFile(path) as source:
        with path.open('r+b') as output:
            output.truncate(source.data_offset + 16)
        with pytest.raises(ValueError, match='short source read'):
            source.tensor_bytes('values', 12, 24)
