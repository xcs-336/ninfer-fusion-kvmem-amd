# Native fusion preparation and qualification

This layer keeps the original public Ops wrappers, packed weight contracts, workspace
ownership, registered shapes and numerical tests. It consumes the complete common HIP
source view **after** the actual Linear preparation. It never runs Linear preparation
again, allocates hidden scratch, repacks weights, substitutes public wrappers or supplies
numerical stubs.

## Shared build hook

After `Linear.cmake`, before Core's final `sync_source_view.py` publication, add:

```cmake
option(NINFER_AMD_FUSED "Build native production fusion families and original suites" OFF)
if(NINFER_AMD_FUSED)
  include("${CMAKE_CURRENT_LIST_DIR}/Fused.cmake")
endif()
```

`NINFER_AMD_BASIC_OPS=ON` and `NINFER_AMD_LINEAR=ON` are required for every selected
family except RMSNormRoPE and GDNGatingProj, which require actual BasicOps.
Only LinearAdd, LinearSwiGLU, GDNInputProj and AttnInputProj consume the shared
integer/hipBLAS auxiliary library. `NINFER_AMD_FUSED_FAMILIES` is a semicolon list.
A first production closure is `linear_add;linear_swiglu;rmsnorm_rope`: 78 translation
units including the three shared integer/hipBLAS implementations. This also creates the
unchanged `ninfer_amd_linear_gguf_test` against the actual public Add and both SwiGLU
parent forms. The BLAS suite additionally requires GDNInputProj and AttnInputProj, so
it is reported pending when those libraries are absent; its oracle is never shortened.
Selecting GDNInputProj when `ninfer_gdn_ops_amd` also exists restores the unchanged
`ninfer_amd_gdn_replay_fold_op_test`, including its convolution snapshot/record calls.
Its implementation dependencies are the actual GDN and GDNInputProj libraries plus
their existing Core/BasicOps/Linear dependencies; it does not require GDNGatingProj.

The legacy Q8 grouped core carries no schedule type, so its fused callers separately
admit their exact `KSplits*(1024+128*TileCols)` allocation and use the same admitted
partition for both the kernel template and launch block size. The unified grouped
schedule has the same allocation; Add, Pair, GDNInput and attention-input callers
apply that admission before instantiating the original schedule. The first complete
shared O3 build exposed the formerly uncovered Add unified C64/W8 (73,728 bytes),
C72/W8 (81,920), C80/W8 (90,112) and C128/W4 (69,632) instances; the production
capacity assertion stays intact. LinearPair C256 in both implementations cannot
fit at the kernel's minimum two K partitions (67,584 bytes); it executes independent
T128 views of the existing tensors, with the original workspace and packed weights.

The Q5 integer-A8 512-token tile also exceeds native LDS (76,800 bytes with its
ring buffers). Q5A8 Add and Q4/Q5 GDN/attention input closures admit their real
`shared_bytes` at compile time and re-enter the whole quantizer-plus-consumer
closure at half NT when necessary (NT8/256 tokens: 43,008 bytes). Thus producer
and every consumer keep the same activation fragment layout, with the original
per-T arena bound and packed weights. The private Q4A8 SwiGLU tile is 20,992
bytes and retains its original physical shape. Integer-A8 policies stay admitted.

All eight families contain 197 native production translation units:

| Family | Native sources |
| --- | ---: |
| LinearAdd | 44 |
| LinearSwiGLU | 32 |
| GDNInputProj | 44 |
| GDNGatingProj | 4 |
| AttnInputProj | 48 |
| LinearPair | 9 |
| LinearTopK | 14 |
| RMSNormRoPE | 2 |

Five original TMA translation units are retained for audit but omitted from native
targets. Their callers select existing complete ordinary LDS/MMA implementations.
NVFP4 uses the separately qualified exact register decode plus native FP16 WMMA
bridge, with K128 staging in place of K256's 73,728-byte staging. The common Linear
overlay must supply its actual staging/output union; the original size assertion is
retained. This is software FP4 decoding followed by native WMMA, not hardware FP4 MMA.
NVFP4 device sources explicitly use `-fhip-fp32-correctly-rounded-divide-sqrt` for
the original scale-encoding RNE contract; no unavailable OCML rounded-operation API
or fake NVIDIA architecture is selected.

GDN's cooperative implementation uses `hip/hip_cooperative_groups.h`,
`hipLaunchCooperativeKernel`, the runtime cooperative capability, and occupancy for
the exact compiled Full/Predicated kernel pair. Per-device cached occupancy replaces
the NVIDIA 2/4/3 CTA constants. Unsupported or insufficient native capacity selects
the original complete unsplit/normalization-composition route before launch.
Normalization kernels recheck their own exact specialization before launch.

## Isolated validation

`standalone/CMakeLists.txt` copies an already prepared view into its own build folder
and imports real previously built dependency archives. It never edits the shared
generated view or invokes the shared Ninja graph. Set `NINFER_AMD_BASE_VIEW`,
`NINFER_AMD_SHARED_BUILD`, `NINFER_PYTHON`, and the common `NINFER_GPU_LOCK_FILE` as
needed. The default family is RMSNormRoPE; configuring/generating this actual graph
succeeded at `research/fused-rms-link`. No object or GPU result follows from configure.

`probe_fused.py` performs one CPU frontend at a time and never launches GPU code.
All 197 native sources passed a first frontend sweep. After the common resource
updates, the final GDN four and TopK fourteen sources plus the three grouped Q8
callers, legacy NVFP4 decode and original GDN gating/replay-fold tests passed a
24-source frontend recheck (`research/fused-resource-final-frontends.json` and
`research/fused-grouped-final-frontends.json`, process terminal exit0). Full production
archive builds, links and original numerical suites remain required. No fused
operator is marked GPU-qualified by source preparation or frontend success.
The subsequent three complete integer-A8 NT admission closures and the independent
Pair C256 boundary probe also passed a four-source frontend batch, terminal exit0
(`research/fused-a8-boundary-final-frontends.json`).

The standalone RMSNormRoPE production archive and unchanged original test executable
subsequently compiled and linked successfully in that isolated folder (O3, one
compiler process). The executable is `ninfer_amd_rmsnorm_rope_op_test.exe`; actual
GPU mathematical execution was performed by the parent under the common GPU lock.
The **complete original RMSNormRoPE suite passed**, terminal exit0, recorded in
`research/fusion-amd-rmsnorm-rope-fixed.log`. This is numerical/behavioral
qualification, with no performance result inferred from the CPU/GPU overlap.

The first GPU run exposed 19 exact split-route mismatches, all in text channels
32..63, while the independent FP64 numerical verdicts passed. Actual O3 IR showed
that BasicOps RoPE orders the second-half addition as `first*sin + second*cos`,
whereas the fused lane-select form ordered it as `second*cos + first*sin`.
Contraction therefore picked a different FMA, occasionally crossing a BF16 rounding
boundary. The production overlay now explicitly rounds `second*cos` and evaluates
`fmaf(first, sin, second*cos)` for that half. Normalization, intermediate BF16
rounding, coefficient calculation and the first-half output remain unchanged.
No assertion, tolerance or reference was relaxed. The actual IR evidence is
`research/rmsnorm-rope-split-before.ll`, `research/rmsnorm-rope-fused-before.ll`,
and `research/rmsnorm-rope-fused-fixed.ll`; the fixed kernel retains native
`llvm.fma.f32` calls with the split path's operand order.

The extra `ninfer_amd_linear_pair_grouped_boundary_test` forces the real C256
candidate through both Legacy and Unified tables at T129/T255/T256. Its 16 nonzero
dyadic activations and stored Q8/power-of-two FP16 scales give exact FP64 sums that
are also representable in FP32, so it checks all output BF16 bits without a loose
tolerance, plus repeated launches, guards and unchanged input bytes. This bounded
probe supplements the original dense public Pair suite and has not run on the GPU.

`audit_oracles.py` compares retained test sources against the common HIP view. It
confirmed all retained test sources keep byte-identical numerical oracles, tolerances, guards,
boundaries and workspace checks. Only GDN's explicitly bounded `sm_count_routes`
hardware-policy namespace changes: its strict resident-grid and fallback assertions
now use native compiled-kernel capacity. Its output-value reference remains unchanged.

The current authoritative Linear resource policy may advance after this frontend
sweep. Regenerate from the newest complete common→Linear view before final builds,
especially after the Q8 K-split 64-KiB resource corrections. No GPU launch is performed
by these preparation, frontend, or oracle-audit tools. The parent task owns the shared
GPU lock and schedules all numerical execution.

## Original test host-memory admission

These are static high-water bounds for explicit host arrays, derived from the
original generators, temporary scopes and snapshot copies. They exclude the HIP
runtime, compiler processes and thread stacks. Recheck available physical memory
before each complete suite, add those costs, and retain the parent's 2-GiB floor.

| Original suite | Explicit host-array budget | Relevant lifetime |
| --- | ---: | --- |
| Combined GGUF | 8 MiB | Full FP32 decode is only 512×1536 (3 MiB); max T130. |
| LinearTopK | 5 MiB | Full vocabulary payload is constructed on the device; hidden is 5120×257×2 bytes and row patches are small. |
| LinearAdd | 0.6 GiB | Patterned weights are packed directly; full-output Q8 T≤8 and Q5 small-T use at most a 5120×17408 FP32 oracle (340 MiB). |
| LinearSwiGLU shared harness | 0.75 GiB | Q8 packed weight is 180.625 MiB, positive/negative FP64 references are 136 MiB each; full weight snapshot and output readback occur in separate scopes. |
| Private W4 BLAS prefill | 1.1 GiB | Largest case K6144/T13057 holds activation, residual and full FP64 output with transient BF16 readback; about 0.92 GiB of explicit arrays. |
| AttnInputProj | 0.35 GiB | BF16 parent 140 MiB plus 140-MiB preservation snapshot and 30-MiB FP32/BF16 activation; snapshots/output planes are checked sequentially. |
| GDNInputProj | 0.25 GiB | Peak FP8 parent 80.031 MiB plus 30-MiB activation and 100-MiB QKV output readback. |

The input-projection helper decodes one FP64 weight row at a time (at most 40 KiB),
and SwiGLU reads the packed weight directly in its FP64 oracle; neither retains a
full FP32 weight matrix. Replay-fold has separate multi-layer record snapshots
and needs its own admission audit before execution. These bounds do not qualify
any numerical suite as passed.
