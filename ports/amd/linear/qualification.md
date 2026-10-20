# Native public Linear qualification

`Linear.cmake` compiles the original public `linear.cpp`, registered format dispatchers, and
independent upstream mathematical test support. The recursive original manifest contains 176
sources: 140 native Linear translation units, 34 existing BasicOps dependencies, and two
NVIDIA TMA-only NVFP4 translation units. Their ordinary LDS/WMMA production routes remain.

The generated source view preserves packed weight formats, public shapes, activation policies,
workspace ownership, original FP64 oracles, comparison criteria, guards, and input-preservation
checks. AMD devices retain the shared `Unmeasured` device class until actual route measurements.
Actual NVFP4 device translation units explicitly retain correctly rounded FP32 division/sqrt
for their activation codec's `__fdiv_rn` boundaries; host-oracle floating-point options remain.

The gfx1201 workgroup LDS limit is 65,536 bytes. Native schedules retain each requested K
partition if it fits; excessive sliced-K partitions use the largest admitted power-of-two K-warp
count. All dependent thread, staging, and reduction dimensions follow that count. FP8 A8
K256 tiles that need 98,304 bytes use K128. The original BF16 TMA selector carries a real
32x32x64, two-stage ordinary LDS/WMMA schedule. Grouped Q8 call sites change both kernel
parameters and block thread counts; one oversized two-stage route uses its supported single
stage. No NVIDIA architecture macro or relaxed mathematical assertion supplies admission.

`resource_probe.cu` checks the actual legacy and unified Q8 storage types across 242 combinations,
including exact storage sizes and loader/thread consistency. The production kernels retain
their original layout assertions, and native 64 KiB assertions also cover dynamic-LDS schedules.

The eleven unchanged original BF16, Q4/Q5/Q6/Q8, T2 A16/A8, FP8 A16/A8, and NVFP4 A16/A4
suites remain registered. Seven bounded public probes additionally exercise actual dispatch,
the original packed generators and FP64 oracle, graph replay, guards, and preserved inputs.
They complement the full suites; their smaller host fixtures do not qualify every shape.

The unchanged combined GGUF test additionally needs the complete public LinearAdd and
LinearSwiGLU libraries. Fused.cmake owns that registration once those dependencies exist.
The original TMA staging-race test still requires its actual mechanism and is not a passing
or skipped stand-in for the native path.

Current evidence (2026-10-20): all 83 resource-sensitive production translation units pass the
native frontend with the actual 64 KiB limit; all 242 storage combinations pass compile-time
checks. The complete native public Linear library, bounded public probe, and all eleven original
family executables compile and link at O3 (`build/build-linear-resource.log`, exit 0).
The parent subsequently ran all ten complete original quantized suites on the RX 9070:
Q4/Q5/Q6/Q8 A16, T2 A16/A8, FP8 A16/A8, and NVFP4 A16/A4 all returned zero. The local
workspace receipt is `research/fusion-amd-linear-originals.json`, with one log per suite;
the observed available physical memory stayed above 2.902 GiB. These are public mathematical
operator results, with the original assertions and full fixture domains. Complete BF16 and
combined GGUF integration remain separate gates; no inference performance advantage follows.

The original quantized suites already avoid full dequantized vocabulary matrices: their packed
generators leave `dequant` empty, their sampled oracle decodes 32 rows, and output scanning uses
a 2 MiB buffer. The static explicit-host-array bounds are Q4 0.879 GiB, Q5 3.188 GiB, Q6
1.159 GiB, Q8 2.538 GiB, T2 A16/A8 0.507 GiB each, NVFP4 A16/A4 0.208/0.887 GiB,
and FP8 A16/A8 2.390/0.354 GiB. Full original suites are preferred when their measured process
overhead and concurrent work fit the physical-RAM admission floor. No additional quantized
vocabulary probe is needed. [Host-memory audit](host_memory.md) records the allocation lifetimes.

The separate BF16 fixture peaks near 4.94 GiB of explicit arrays because its preservation check
duplicates the complete two-byte vocabulary payload while 200 MiB of outer weights remain live.
Its weight reference is not a full FP32 matrix either. The native source-view preparer now adapts
only this test helper to use at most 2 MiB of heap scratch and to compare every uint16 in consecutive
readback intervals, including the final partial interval. The existing message/failure, every case,
FP64 oracle, and byte criterion remain. This reduces the static explicit-host bound near 2.57 GiB.
The isolated actual HIP header passes the CPU frontend; its complete original GPU suite still
requires qualification after the shared build publishes the new helper.
