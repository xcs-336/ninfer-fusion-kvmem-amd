# gfx1201 BF16 MMA candidate

Status: **CPU layout checks passed; HIP compilation and GPU execution NOT RUN here.**
This is additive collaboration material. Neither baseline's existing source is replaced.
There is no claim of Engine support, production Op correctness, speed or memory reduction.

`mma_bf16_gfx12.h` implements the BF16/FP32 `m16n8k16.row.col` fragment operation with
the native gfx12 wave32 `16x16x16` BF16 WMMA builtin and register shuffles. It does not
translate PTX at runtime and does not use a CUDA compatibility runtime. The fixed
fragment boundary is a correctness bridge; production AMD GEMM should load native
WMMA layouts directly to avoid these expensive redistributions and unused columns.

All 32 lanes must participate with no divergence; compile only for gfx1201 wave32.
The same signature can be wrapped from `ninfer::ops::common::mma_bf16` in upstream or
`ninfer::ops::mma_bf16` in ninfer-all AFTER standalone qualification. The existing
`ldmatrix`, synchronization and surrounding kernels still require separate work.
Do not include the PTX implementation in a HIP translation unit.

## CPU qualification

The checked local interpreter is Python 3.13.5; this standalone test uses only the
standard library. It does not change the project's maintained Python environment.

```powershell
python ports/gfx12-mma/test_layout.py
# Optional independent comparison with AMD's calculator (needs its tabulate dependency):
python ports/gfx12-mma/test_layout.py --calculator ../amd-matrix-reference/matrix_calculator.py
```

The second command checked 1024 official A/B/C/D positions and 386 integer product
cases. The layout model is not execution of the C++ header and cannot establish its
compilability. AMD calculator pin:
`ROCm/amd_matrix_instruction_calculator@2ef91896bcdc4d26624f952e5c905c787cd9bc9e`.

## Hardware qualification (commands prepared, not executed here)

Run from the root of the chosen source tree on a free RX 9070. Select the AMD
toolchain explicitly; do not change drivers or invoke the existing NVIDIA GPU.
If hipcc is not in PATH, substitute the installed SDK's full path.

```powershell
New-Item -ItemType Directory -Force .agent-tmp/amd-mma | Out-Null
hipcc --version
hipcc -x hip -std=c++20 -O2 -fno-fast-math -ffp-contract=off --offload-arch=gfx1201 -mwavefrontsize32 ports/gfx12-mma/test_mma.hip -o .agent-tmp/amd-mma/test_mma.exe
if ($LASTEXITCODE -ne 0) { throw 'HIP compilation failed' }
& .agent-tmp/amd-mma/test_mma.exe 2>&1 | Tee-Object .agent-tmp/amd-mma/gpu-results.txt
if ($LASTEXITCODE -ne 0) { throw "Hardware qualification incomplete: exit $LASTEXITCODE" }
```

For Linux, the same hipcc flags apply; choose an output without `.exe` and check its
exit status. Do not pass `-ffast-math`. A compiler without the gfx12 builtin fails
explicitly. Exit 77 means **NOT RUN**, never PASS. The harness uses HIP device 0;
set `HIP_VISIBLE_DEVICES` explicitly before invocation if the RX 9070 is not device 0.

The harness prepares 2098 tiles: 2050 exact cases (C-only, every i/j/k unit product,
one dense integer tile) plus 48 varied BF16 tiles. It runs each with 32/128 threads
per block and 1/3 repeated MMAs. The 128-thread mode also has two inactive final
waves, exercising uniform early return. Each run checks all 128 outputs per tile,
exact cases, FP64 forward error and per-tile relative L2, output guards and input
preservation. The GPU oracle never calls the candidate and never rounds the FP64
expected output to FP32/BF16. Finite normal-or-zero BF16 inputs only are qualified;
denormals, overflow and NaN/Inf semantics require further tests before broad support.

Memory output distinguishes exact simultaneously owned buffer bytes from sampled
whole-device use; samples include other processes and omit unobserved transient
peaks. This is not Engine peak VRAM. Record actual graphics-driver version separately
because `hipDriverGetVersion` is an API version and does not identify Adrenalin.

## Layout rationale and sources

Let `l` be a PTX lane, `g=l/4`, `t=l%4`, and `h=0,1` a halfword. PTX A word `r`
holds `(row=g+8*(r%2), k=2*t+8*(r/2)+h)`; B word `r` holds
`(k=2*t+8*r+h, col=g)`; C register `r` holds
`(row=g+8*(r/2), col=2*t+r%2)`.

For AMD lane `L`, word `r` and halfword `h`, A/B use
`k=8*(r/2)+4*(L/16)+2*(r%2)+h`, with A row `L%16` and B col `L%16`.
C/D register `r` holds `(row=8*(L/16)+r, col=L%16)`.
Set B/C cols 8..15 to zero. Preserve the incoming C in WMMA; adding C after a
zero-C WMMA introduces a different FP32 rounding boundary. No bitwise parity with
NVIDIA tensor cores is promised for general floating-point inputs.

A frequent wrong implementation selects a register based on the destination lane
and then shuffles it. That selection executes in the source lane and can select
the wrong register. The candidate shuffles each uniform register first and then
selects; every lane executes every shuffle, including padded lanes.

- [NVIDIA PTX m16n8k16 floating-point fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)
- [AMD calculator pinned source](https://github.com/ROCm/amd_matrix_instruction_calculator/blob/2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py)
- [LLVM gfx12 builtin declaration](https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td)

The LLVM main declaration was inspected on 2026-10-20. It is supporting evidence,
not proof that a particular installed clang has that builtin; compilation checks it.
