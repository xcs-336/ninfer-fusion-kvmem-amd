"""CPU-only fragment transport checks; never executes or qualifies GPU code."""

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys


def ptx_a(lane, word, half):
    return lane // 4 + 8 * (word % 2), 2 * (lane % 4) + 8 * (word // 2) + half


def ptx_b(lane, word, half):
    return 2 * (lane % 4) + 8 * word + half, lane // 4


def ptx_c(lane, reg):
    return lane // 4 + 8 * (reg // 2), 2 * (lane % 4) + reg % 2


def amd_ab(lane, scalar, matrix):
    k = 8 * (scalar // 4) + 4 * (lane // 16) + scalar % 4
    return (lane % 16, k) if matrix == "A" else (k, lane % 16)


def amd_cd(lane, reg):
    return 8 * (lane // 16) + reg, lane % 16


# The proposed device code's shuffle routes, independently checked against
# coordinates from the published PTX and AMD layouts above.
def gather_a(lane, word):
    row = lane % 16
    return 4 * (row % 8) + 2 * (lane // 16) + word % 2, row // 8 + 2 * (word // 2)


def gather_b(lane, word):
    return 4 * (lane % 8) + 2 * (lane // 16) + word % 2, word // 2


def gather_c(lane, reg):
    return 4 * reg + (lane % 16) // 2, 2 * (lane // 16) + lane % 2


def scatter_d(lane, reg):
    return 16 * (reg // 2) + 2 * (lane % 4) + reg % 2, lane // 4


def check_transport():
    for func, words, halves, expected in ((ptx_a, 4, 2, 256), (ptx_b, 2, 2, 128)):
        assert len({func(l, w, h) for l in range(32) for w in range(words)
                    for h in range(halves)}) == expected
    assert len({ptx_c(l, r) for l in range(32) for r in range(4)}) == 128
    count = 0
    for lane in range(32):
        for word in range(4):
            al, ar = gather_a(lane, word)
            bl, br = gather_b(lane, word)
            for half in range(2):
                assert ptx_a(al, ar, half) == amd_ab(lane, 2 * word + half, "A")
                if lane % 16 < 8:
                    assert ptx_b(bl, br, half) == amd_ab(lane, 2 * word + half, "B")
                count += 1
        for reg in range(8):
            if lane % 16 < 8:
                cl, cr = gather_c(lane, reg)
                assert ptx_c(cl, cr) == amd_cd(lane, reg)
        for reg in range(4):
            dl, dr = scatter_d(lane, reg)
            assert amd_cd(dl, dr) == ptx_c(lane, reg)
    return count


def calculator_check(path):
    # Read the official calculator's CSV, not our inverse formulas. Four independent
    # matrices cover every 16-bit input location and every FP32 accumulator location.
    checks = 0
    for matrix in "ABCD":
        cmd = [sys.executable, str(path), "-a", "gfx1201", "-i",
               "v_wmma_f32_16x16x16_bf16", "-w", "32", "-M", "-" + matrix, "-c"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("lane,"))
        rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
        assert len(rows) == 33, (matrix, len(rows))
        for row in rows[1:]:
            lane = int(row[0])
            assert len(row) == 9
            for reg, observed in enumerate(row[1:]):
                i, j = amd_ab(lane, reg, matrix) if matrix in "AB" else amd_cd(lane, reg)
                assert observed == f"{matrix}[{i}][{j}]", (matrix, lane, reg, observed, i, j)
                checks += 1
    return checks


def model(a, b, c):
    pa = [[[a[ptx_a(l, w, h)[0]][ptx_a(l, w, h)[1]] for h in range(2)]
           for w in range(4)] for l in range(32)]
    pb = [[[b[ptx_b(l, w, h)[0]][ptx_b(l, w, h)[1]] for h in range(2)]
           for w in range(2)] for l in range(32)]
    pc = [[c[ptx_c(l, r)[0]][ptx_c(l, r)[1]] for r in range(4)] for l in range(32)]
    wa = [[0] * 16 for _ in range(16)]
    wb = [[0] * 16 for _ in range(16)]
    wc = [[0] * 16 for _ in range(16)]
    for lane in range(32):
        for word in range(4):
            al, ar = gather_a(lane, word)
            bl, br = gather_b(lane, word)
            for half in range(2):
                i, k = amd_ab(lane, word * 2 + half, "A")
                wa[i][k] = pa[al][ar][half]
                k, j = amd_ab(lane, word * 2 + half, "B")
                wb[k][j] = pb[bl][br][half] if j < 8 else 0
        for reg in range(8):
            i, j = amd_cd(lane, reg)
            cl, cr = gather_c(lane, reg)
            wc[i][j] = pc[cl][cr] if j < 8 else 0
    wd = [[wc[i][j] + sum(wa[i][k] * wb[k][j] for k in range(16))
           for j in range(16)] for i in range(16)]
    out = [[None] * 8 for _ in range(16)]
    for lane in range(32):
        for reg in range(4):
            dl, dr = scatter_d(lane, reg)
            wi, wj = amd_cd(dl, dr)
            i, j = ptx_c(lane, reg)
            out[i][j] = wd[wi][wj]
    return out


def check_products():
    dense_a = [[(i * 7 + k * 3) % 17 - 8 for k in range(16)] for i in range(16)]
    dense_b = [[(k * 5 + j * 11) % 19 - 9 for j in range(8)] for k in range(16)]
    c = [[i * 8 + j - 64 for j in range(8)] for i in range(16)]
    cases = [(dense_a, dense_b), ([[0] * 16 for _ in range(16)], dense_b)]
    for i in range(16):
        for k in range(16):
            a = [[0] * 16 for _ in range(16)]
            a[i][k] = 1
            cases.append((a, dense_b))
    for k in range(16):
        for j in range(8):
            b = [[0] * 8 for _ in range(16)]
            b[k][j] = 1
            cases.append((dense_a, b))
    for a, b in cases:
        expected = [[c[i][j] + sum(a[i][k] * b[k][j] for k in range(16))
                     for j in range(8)] for i in range(16)]
        assert model(a, b, c) == expected
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculator", type=Path)
    args = parser.parse_args()
    locations = check_transport()
    official = calculator_check(args.calculator) if args.calculator else None
    cases = check_products()
    print(json.dumps({"scope": "CPU layout and integer algebra only", "status": "PASS",
                      "a_locations": locations, "official_locations": official,
                      "integer_product_cases": cases, "gpu_compile": "NOT_RUN",
                      "gpu_execution": "NOT_RUN"}))


if __name__ == "__main__":
    main()
