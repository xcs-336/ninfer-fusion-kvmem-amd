# Original Linear suite host-memory audit

Static audit, 2026-10-20. These are conservative high-water bounds for explicit live host
arrays in the current original fixtures, not measured process working sets. Add runtime,
driver, executable, and thread overhead before applying a physical-RAM admission floor.

| Original suite | Explicit host arrays, GiB | Dominant case |
|---|---:|---|
| Q4 A16 | 0.879 | `[4304,1152]`, T=131072, BF16 input plus FP32 activation oracle |
| Q5 A16 | 3.188 | `[1152,4304]`, T=131072, BF16 input plus FP32 activation oracle |
| Q6 A16 | 1.159 | `[1152,1536]`, T=131072, BF16 input plus FP32 activation oracle |
| Q8 A16 | 2.538 | `[248320,5120]`, packed payload plus complete preservation readback |
| T2 A16 | 0.507 | `[5120,17408]`, full-row FP32 weight oracle and output comparison |
| T2 A8 | 0.507 | Same full-row fixture; these original T2 suites do not make graph copies |
| NVFP4 A16 | 0.208 | `[34816,5120]`, packed payload plus complete preservation readback |
| NVFP4 A4 | 0.887 | `[34816,5120]`, T=128, full-row oracle and preservation readback |
| FP8 A16 | 2.390 | `[248320,5120]`, packed payload plus complete preservation readback |
| FP8 A8 | 0.354 | `[34816,5120]`, packed payload plus complete preservation readback |
| BF16 A16, original helper | 4.940 | BF16 vocabulary bits plus same-sized preservation copy, with outer 200 MiB weights live |
| BF16 A16, native helper | 2.570 | Same original fixture with complete 2 MiB chunked preservation readback |

`linear_test_common.cpp::run_shape` uses `make_patterned_weight`, which directly generates the
packed payload without allocating a source or dequantized matrix. `PackedWeight::dequant` stays
empty in this path. `SampledRows` materializes only 32 rows for its independent FP64 oracle;
every token column remains covered. `read_output` scans all output words in chunks of at most
1,048,576 BF16 elements (2 MiB). Therefore a whole `N*K*sizeof(float)` weight matrix is **not**
the allocation made by the large quantized vocabulary cases.

For packed payload P bytes, selected rows R, K inputs, and maximum token extent T, the principal
live arrays are `P + 4*R*K + 2*K*T + 8*R*T`. The temporary full activation oracle adds `4*K*T`.
Output selection and index arrays together add up to `16*R*T`, plus the 2 MiB scan buffer.
Input preservation finally keeps the original payload alongside a complete P-byte readback,
and the original activation alongside its BF16 readback. The table takes the maximum lifetime
stage per case, rather than adding arrays that have already been destroyed.

BF16 uses a separate fixture. It directly stores two-byte BF16 weights and decodes their values
on demand for its FP64 dot products. Its vocabulary peak is caused by `DeviceWeight::verify_preserved`
allocating a second complete BF16 weight buffer; there is no full FP32 weight oracle there either.

With 6.63 GiB physically available, reserving 0.5 GiB for process/runtime overhead and a 2 GiB
remaining floor permits the ten quantized suites one at a time under this static budget.
BF16 needs about 7.44 GiB available with the same allowances. Remeasure immediately before
admission, and budget any concurrent compiler or model process separately. These allowances
do not substitute for observing the actual child working set and available physical memory.

No additional quantized vocabulary probe is needed: the unchanged Q6, Q8, and FP8 suites already
exercise real N=248320 with bounded sampled-row oracles. If BF16 still cannot be admitted after
cleanup, a fixture-only chunked preservation readback can retain every original case, full-row
T=1 FP64 oracle, and byte comparison while reducing its explicit peak to about 2.57 GiB.
That test-helper adaptation is now implemented in the generated native view and passes isolated
host frontend checking. Full original BF16 GPU qualification remains pending after rebuild.

The generated-helper body uses one heap vector of at most 1,048,576 uint16 elements,
iterates consecutive intervals `[begin, begin+count)` over `host.bits`, synchronously copies the
same interval from `device.p`, and compares every copied uint16 with its original host element.
It returns the existing failure and message on the first difference and returns zero only after
all intervals match. No interval is sampled or omitted, including the final partial interval.
The helper remains outside graph capture, after the existing stream synchronization. A stack
array is inappropriate because a 2 MiB automatic buffer can exceed the Windows thread stack.
