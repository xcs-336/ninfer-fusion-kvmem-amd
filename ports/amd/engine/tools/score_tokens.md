# Teacher-forced token score export

`score_tokens.cpp` is a research consumer of the unchanged public Engine. The
implementation has compiled and linked on Windows / gfx1201. All17 actual native
help/strict-parser/default-plan checks passed in the guarded28-case receipt
`research/ninfer-frontend-fixtures-native-first/receipt.json`; they include invalid
first-target boundaries, JSON number types/shape/trailing data, duplicate/unknown
options, short context and preservation of an existing output. The plan retained
the19 actual frontend IDs and constructed no Engine. Together with11 frontend
runs, minimum available RAM was4.151GiB.

The real model-scoring path then executed on2026-10-20 at12:44 UTC, using all23
actual fixed IDs with first_target19, BF16 KV/context512, FP32 state, A16 and
profile off. `research/ninfer-native-score-first/receipt.json` records
`SCORES_COMPUTED_RESOURCE_QUALIFIED`, exit0 and four complete finite natural-log
scores, including the given EOS target. Host guard minimum was3.382GiB; the
two PID monitor rows reported a3533.875MiB host minimum and one non-null adapter
headroom sample of7650.082MiB, with no guard event. This is a bounded scoring
execution and input/domain/resource check, not a paired quality result or a
performance advantage. Raw values and complete input remain in
`research/ninfer-native-score-first/native-scores.json`.

The input is one JSON array of nonnegative int32 token IDs. It must contain at
least two IDs, at most `--context` IDs, and fit both the 64 MiB text-file limit
and the context-sized `32*context+4096` byte limit. Floating-point JSON numbers,
strings, booleans, comments, objects, trailing data, negative/out-of-int32 IDs,
duplicate CLI options and unknown options are rejected. No tokenization,
template, implicit BOS/EOS, truncation or stop handling occurs. All IDs, including
special/EOS IDs in the middle, are passed to the single public scoring call.

`first_target` is a zero-based position, `1 <= first_target < token_ids.size()`.
For each position `i` from that boundary through the final input ID, the output
is the natural logarithm `log p(T[i] | T[0],...,T[i-1])`. The entire preceding
prefix is retained. Each call uses the original fresh-State/Main-KV scoring
route, rather than generation or a reimplemented logit/softmax path.

Default invocation checks CLI/JSON/paths and prints a `plan_only` JSON document
to stdout. It constructs no Engine and writes no result file. `--run` performs
the model operation and must be launched by the owner under the existing GPU
lock and host/GPU memory guard. This tool itself does not acquire those resources.
The output path and its `.partial` sibling must both be new. Only a complete,
validated result is renamed to the final path; errors return nonzero and produce
an error JSON on stderr.

The fixed execution profile is CausalScoring, FP32 GDN state, A16 activations,
no speculative backend, no CUDA Graph, no prefix cache, no Vision, no load-time
head/embedding/expert transcoding, native RoPE, and device-profile `off`. KV and
device/context remain explicit choices, with defaults bf16/device0/context512.
The original Engine normalizes prefill_chunk to1024, max_concurrency to1,
max_pending_requests to1 and explicit KV capacity to max_context; the tool
verifies the effective options before scoring and records them.

The original startup sink/ring guard also applies to CausalScoring: there is no
purpose exception in `program/planning/startup.cpp:1056–1129`. With the default
sink4 pages plus the required4 working pages, the explicit pool needs8*64=512
tokens. `sink_pages_for_guard():108–113` supplies the default4 pages. Therefore
the default is512, matching the accepted native smoke/reference; setting256
under the unchanged default sink will be refused by the original Engine.
The tool neither overrides that environment variable nor weakens the guard.

## Result schema version1

Every successful model result has these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Integer1 |
| `status` | `computed`; this is no quality/PASS designation |
| `log_base` | `e`, fixed |
| `token_ids` | Complete ordered input array, including the context-only prefix |
| `token_count` | Complete input length |
| `first_target` | First scored zero-based input position |
| `scores` | Ordered array of `{position,target_id,logprob}` for every scored position |
| `scored_tokens` | `token_count-first_target`, exactly `scores.size()` |
| `public_token_count` | Contiguous public tokenizer domain count |
| `physical_vocab_size` | Current public ModelMetadata.vocab_size implementation: model config rows |
| `logprob_vocabulary` | `{first_token_id:0,token_count:public_token_count}` |
| `domain_provenance` | How public/physical counts and normalizer scope were established |
| `engine` | Actual public LoadSummary/ModelMetadata, with their source named |
| `declared_identity` | Caller comparison/model/weights/tokenizer/engine labels, explicitly unverified |
| `execution` | Effective normalized precision and Engine options |
| `total_nll`, `mean_nll` | FP64 accumulation over the returned FP32 logprobs |
| `input_semantics` | No retokenization/special insertion/stop/window truncation, fresh state/KV |
| `artifact_path`, `input_ids_path`, `output_path` | Absolute paths for sidecar provenance |

No NaN/infinity or positive logprob is accepted, and no partial target list is
published. Float results are promoted to double for JSON serialization without
decimal rounding. No pass threshold or additional exponentiation is imposed.

Current ModelMetadata assigns physical config rows to `vocab_size`, even though
its header comment calls it the tokenizer domain. The tool keeps that field
unchanged and obtains the public domain through `Engine::token_bytes`: id0 must
be legal and id=physical_rows must be illegal, then the contiguous valid prefix
is located by binary search. Only the exact original frontend
`std::out_of_range("token id is outside the tokenizer vocabulary")` is a negative
answer; all other errors propagate. The original loader already requires a
contiguous public domain. Complete input IDs are checked against that boundary
before score_tokens, which retains its own original validation.

The production scoring normalizer is `target_logprobs` over
`model.resources().public_token_count`, not all physical output rows. Its kernel
loops only while `row < valid_rows`. Thus Bonsai's 248,320-row head and its
248,077-token public vocabulary remain distinct in the export. A comparator must
require equal complete token arrays, first_target and logprob_vocabulary before
comparing numerical errors; it cannot equate raw full-physical-domain logprobs.

Caller `--model-id`/`--weights-id` labels and public ModelMetadata report values
are not certified artifact-header identity. Exact artifact/model/weights identity
and input/artifact SHA256 belong to the existing independently verified Python
sidecar. The C++ program never hashes or reads the full weight file itself.

## Owner CMake wiring

Root owns all CMake edits. A prospective target is:

```cmake
add_executable(ninfer_amd_score_tokens
  "${CMAKE_CURRENT_LIST_DIR}/engine/tools/score_tokens.cpp")
ninfer_native_hip(ninfer_amd_score_tokens)
target_link_libraries(ninfer_amd_score_tokens PRIVATE ninfer_engine_amd ninfer::json)
ninfer_product_stage_runtime(ninfer_amd_score_tokens)
```

This uses the real Engine closure and the existing product runtime-DLL staging.
It needs C++20 and the existing json interface target; it introduces no new GPU
kernel, alternate model loader or production API. Place the wiring after the
existing `ninfer_product_stage_runtime` function is defined. Do not register an
automatic model/GPU CTest. CPU-only help/invalid-input/plan checks can run after
the owner builds the target; actual scores require a separate locked admission.

Example future invocation, shown without execution:

```powershell
ports/amd/build/ninfer_amd_score_tokens.exe `
  --artifact D:/programfile/work/deepseek/artifacts/ninfer-amd/bonsai2-pq2-mtp-q8.ninfer `
  --tokens D:/programfile/work/deepseek/research/quality/input-token-ids.json `
  --first-target 1 --context 512 --kv-dtype bf16 --device 0 `
  --comparison-id bonsai-native-reference `
  --model-id bonsai2-27b-pq2-mtp `
  --weights-id bonsai2-pq2_0-mtp-q8_0-amd-native-v1 `
  --output D:/programfile/work/deepseek/research/quality/native-scores.json
```

Add `--run` only inside the owner's established lock/guard launcher. Plan success
or a JSON fixture is not a successful model score and must not become E2E PASS.
