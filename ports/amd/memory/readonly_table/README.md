# Ordinary BF16 pointer-table load candidate

2026-10-21: ACTUAL SIX-CASE FUNCTIONAL PASS, DEFAULT OFF, NOT ENGINE OR PERFORMANCE
QUALIFIED. The actual target built, passed its strict link map and the same six cases,
18 same-Graph replays and 1,259,520 output checks as the baseline. Thirty-two actual
paired kernel bodies were extracted from research copies; no build object was modified.

`Core.cmake` includes `SegmentedBF16Readonly.cmake` after `SegmentedPool.cmake` and
`SegmentedBF16Ops.cmake`. An owner-controlled configure must explicitly enable
`NINFER_AMD_SEGMENTED_BF16_READONLY`. This creates
the excluded-from-all target `ninfer_amd_segmented_bf16_readonly_test`; it registers no
automatic GPU test. The target must be built explicitly and executed only under the
current root window, lock, host guard and PID resource-monitor protocol.

The new generator pins all five baseline BF16 outputs, its original helper/overlay/
launcher/leaf, the frozen original fixture and criterion, both map verifiers, the
isolated Core outputs and their source receipts. It writes a new content-addressed
include root exclusively, verifies byte-identical existing output, and refuses changed
baselines. It never overwrites the old source view.

The only semantic difference is:

```cpp
const auto address = page_bases[cell];
```

in place of the frozen helper's volatile U64 cast/read. Two explanatory comment lines
also change. All address arithmetic, kernels, wrappers, math, six-case fixture, compile
flags, allocator and Graph contracts remain the same. This does not add shuffle,
pointer staging or a concurrent-publication contract.

Publication must finish before a reader begins, the table epoch must remain fixed
through its entire dispatch/replay, and all old readers must end before relocation.
Ordinary reads do not relax those requirements. A map gate proves the actual archive
closure, not the opcode for an inline helper; actual SDK ISA plus unchanged strict
mathematical/byte/resource checks are still required.

The detailed read-only audit and later page-reuse candidates are in
`research/ninfer-kv-segmented-table-load-audit.md`. No speedup is claimed.


Actual ISA: 16 instances have two fewer static s_wait_dscnt instructions, 14 have
four fewer and two have unchanged counts. VGPR/SGPR/LDS/private/spill fields match
the baseline in all32 pairs. Real selected small-T/prompt kernels still spill;
no zero-scratch, timing, occupancy or E2E benefit is claimed. Full load-dependency
and build-freshness manual review remains separate. Actual host guard minimum
4.277GiB is an admission observation, not a continuous peak or saving.
