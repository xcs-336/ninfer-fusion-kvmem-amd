"""Independent finite address/coverage audit of the native MoE CTA geometry.

This does not execute or qualify floating-point math. The packed FP64 oracle
calls the actual production phase separately.
"""
from collections import Counter
import json


K = 2560
PAIRS = 640
CODE_ROW = K // 2
SCALE_GROUPS = K // 16
SCALE_ROW = 164


def cooperative(rows, width, transfer, threads):
    visits = Counter()
    for tid in range(threads):
        for start in range(tid * transfer, rows * width, threads * transfer):
            assert start % transfer == 0
            assert start // width == (start + transfer - 1) // width
            visits.update(range(start, start + transfer))
    assert visits == Counter({i: 1 for i in range(rows * width)})


def scale_loads(rows, threads):
    visits = Counter()
    for tid in range(threads):
        for item in range(tid, rows * 40, threads):
            row, tile = divmod(item, 40)
            start = row * SCALE_ROW + tile * 4
            visits.update(range(start, start + 4))
    expected = Counter({row * SCALE_ROW + column: 1
                        for row in range(rows) for column in range(SCALE_GROUPS)})
    assert visits == expected  # four-byte row padding is never consumed


def bank_scale(row, group):
    return (row // 128 * 40 + group // 4) * 512 + row % 32 * 16 + row % 128 // 32 * 4 + group % 4


def packed_addressing():
    # The K16M128x4 source plane is a permutation, including gate/up row borders.
    addresses = {bank_scale(row, group) for row in range(1280) for group in range(160)}
    assert addresses == set(range(1280 * 160))
    rows = Counter()
    for block in range(PAIRS // 8):
        pair_base = block * 8
        for local_row in range(16):
            global_row = pair_base + (local_row & 7) + (PAIRS if local_row >= 8 else 0)
            rows[global_row] += 1
            # Every original global row contributes all 1280 code bytes and
            # all 160 stored scales; the local row spacing is only LDS padding.
            for tile in range(40):
                source = (global_row // 128 * 40 + tile) * 512 + global_row % 32 * 16 + global_row % 128 // 32 * 4
                assert [source + part for part in range(4)] == [bank_scale(global_row, tile * 4 + part) for part in range(4)]
    assert rows == Counter({row: 1 for row in range(1280)})


def output_coverage(tokens, token_chunk, waves):
    visits = Counter()
    for block in range(PAIRS // 8):
        for chunk in range(0, tokens, token_chunk):
            for wave in range(waves):
                for lane in range(32):
                    pair = block * 8 + (lane >> 2)
                    for part in range(2):
                        token = chunk + wave * 8 + 2 * (lane & 3) + part
                        if token < tokens:
                            visits[token, pair] += 1
    assert visits == Counter({(token, pair): 1 for token in range(tokens) for pair in range(PAIRS)})


def reader_bounds(activation_rows, waves):
    for wave in range(waves):
        for lane in range(32):
            a_matrix = lane >> 3
            a_row = (lane & 7) + ((a_matrix & 1) << 3)
            a_col = (a_matrix >> 1) * 16
            b_row = wave * 8 + (lane & 7)
            b_col = ((lane >> 3) & 1) * 16
            sfa_row = ((lane & 1) << 3) | (lane >> 2)
            sfb_row = wave * 8 + (lane >> 2)
            for k64 in range(40):
                # ldmatrix address providers each supply an aligned 16-byte
                # row segment; the already-qualified primitive handles lanes.
                assert 0 <= a_row < 16 and k64 * 32 + a_col + 16 <= CODE_ROW
                assert 0 <= b_row < activation_rows and k64 * 32 + b_col + 16 <= CODE_ROW
                assert 0 <= sfa_row < 16 and k64 * 4 + 4 <= SCALE_GROUPS
                assert 0 <= sfb_row < activation_rows and k64 * 4 + 4 <= SCALE_GROUPS


def main():
    reports = []
    packed_addressing()
    for name, rows, buffers, threads, chunk in [('legacy', 8, 1, 32, 8), ('async', 8, 2, 32, 8), ('v2', 16, 1, 64, 16)]:
        cooperative(16, CODE_ROW, 16, threads)
        cooperative(rows, CODE_ROW, 16, threads)
        scale_loads(16, threads)
        scale_loads(rows, threads)
        reader_bounds(rows, threads // 32)
        # All constituent byte sizes are already multiples of alignas(16).
        lds = 16 * (CODE_ROW + SCALE_ROW) + buffers * rows * (CODE_ROW + SCALE_ROW)
        assert lds <= 65536
        for tokens in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65]:
            output_coverage(tokens, chunk, threads // 32)
        if buffers == 2:
            for count in [1, 7, 8, 9, 17, 65]:
                for current in range(0, count, 8):
                    assert ((current >> 3) & 1) != (((current + 8) >> 3) & 1)
        reports.append({'schedule': name, 'lds_bytes': lds, 'address_coverage': 'PASS'})
    print(json.dumps({'production_math_qualification': 'separate GPU FP64 oracle required', 'schedules': reports}, indent=2))


if __name__ == '__main__':
    main()
