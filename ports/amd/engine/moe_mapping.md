The Flash-Next native overlay retains every original model translation unit,
the stored NVFP4 expert banks, all input/intermediate quantizers, routing and
the complete down projection. It changes the three gate/up CTA schedules that
originally requested 69,312 or 92,416 bytes of dynamic shared memory.

| Schedule | CTA threads | Gate/up pairs per CTA | Tokens per chunk | Actual aligned LDS |
|---|---:|---:|---:|---:|
| Legacy | 32 | 8 | 8 | 34,656 B |
| Async | 32 | 8 | 8 | 46,208 B |
| V2 | 64 | 8 | 16 | 46,208 B |

`NativeGateUpLds` describes the actual production allocations and asserts their
`sizeof` values and the 65,536-byte limit. The async schedule retains the
original two activation buffers, commit/wait sequence and barriers. V2 retains
two token groups, now one per wave. A CTA covers eight gate rows and the same
eight up rows at the original +640 global row displacement.

The global code row stride stays 1,280 bytes. Stored scales retain the original
K16M128x4 permutation and 204,800-byte expert plane. The local scale row stride
stays 164 bytes, including four bytes of padding. Forty original K64 MMA steps
contract the full K=2,560 in the original order. Expert divisors, token/path
maps, SiLU multiplied by up, and the observable BF16 activation cast remain
unchanged. `NINFER_FLASH_NEXT_MOE_STAGING` keeps its original legacy/v1/v2
selection semantics.

HIP retains the original finite staging reentry only when
`weights.expert_gate_up.on_host` or `weights.experts_gguf` is true. The original
`slice_width` remains eight with the wide arm off, or the original clamped
8..32 wave width with it on. Each input/output token view is processed in
ascending order on the same stream and with the same workspace scope. The
original expert-slot/metadata capacity checks and the GGUF wide-arm refusal
remain active. Fully resident NVFP4 banks bypass this staging restriction and
retain their complete token domain, including the >=512-token native MMA arm.

`audit_moe_geometry.py` independently enumerates cooperative byte ownership,
the complete stored-scale permutation, every global gate/up row, output
ownership for twelve token/tail domains, LDS reader bounds, and double-buffer
alternation. All three schedules pass this finite address audit. This is
resource/address evidence; it does not establish GPU numerical correctness.

`test_moe_gate_up.cpp` calls `native_flash_next_moe_gate_up_launch`, the same
complete production phase invoked by the full model launcher. It uses four
truthfully described expert-bank slots, three permuted active experts, all
640 output pairs, T=1/9/17, nontrivial token/path maps and all three schedules.
Its independent host oracle decodes stored E2M1 nibbles and E4M3 scales, computes
naive full-K FP64 gate/up dot products and SiLU times up, and checks BF16 outputs,
untouched paths, byte guards and all immutable inputs. The represented phase
inputs are packed activation codes/scales; conversion from BF16 into those
inputs remains the original production quantizer, outside this phase oracle.

The four original MoE implementation TUs have one library owner,
`ninfer_flash_next_moe_kernels_amd`, which is also linked into the full model
runtime. This partition enables a bounded production qualification before the
whole Engine is linked. The original manifest still lists all runtime TUs.

The actual four-TU production closure compiled and linked at O3. The root's
serialized RX 9070/gfx1201 run passed all nine production phase cases and
124,800 BF16 outputs against the packed FP64 oracle, including immutable inputs,
untouched paths and guards (`research/fusion-amd-native-batch-moe-q6.log`).
That phase qualification does not qualify the unchanged full router/grouping/quantization/down
pipeline, optional expert staging, or end-to-end `.ninfer` inference. Those
require their own applicable original Engine tests and a real artifact. The
three direct BF16 schedules in the shared-expert TU use the actual Linear
schedule header and require 49,152 bytes; their mathematical qualification is
also separate from this gate/up resource audit.
