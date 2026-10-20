# CPU frontend export source handoff

Current qualification (2026-10-20): the actual Windows/gfx1201 target compiled and linked. All11 frontend fixture runs and all17 score-tool help/parser/plan checks then passed under one shared byte-0 lock/job and host guard; the complete28-case receipt is `research/ninfer-frontend-fixtures-native-first/receipt.json` (12:38:46–12:38:56 UTC). Minimum observed available RAM was4.151GiB, with fixture0.5/runtime1/floor2GiB. The frontend runs cover English, Chinese, Unicode/NUL/CRLF/partial-UTF8 token pieces, literal control strings, multiturn, thinking on/default/off, exact stop dedup and explicit tool/media/padded-stop rejection. The English prompt retains all19 real IDs, public248077/physical248320 and defaults248046/248044; on/default both produced59 IDs. Rendering replay matched actual prepare for every admitted case. The earlier single-fixture4.700GiB result remains historical evidence. These are CPU frontend/CLI qualifications; Prism alignment and actual model scoring remain pending. This tool does not load model weights, and no model-quality/performance conclusion follows.

## Production route and boundaries

`frontend_export.cpp` includes the generated `models/qwen3_5/frontend/frontend.cpp` in its translation unit. This exposes that exact implementation's anonymous-namespace `compile_chat_template`, `convert_messages`, `render_options`, and `merge_stop_policy` to the research tool without copying them or changing any production API. The authoritative IDs come from real `Frontend::prepare` and `FrontendTestAccess::inspect`. A second call of those same helpers and `encode_rendered_chat` must produce an identical complete ID vector and `starts_in_reasoning`; otherwise the tool exits nonzero before writing success JSON.

The tool uses real `product::prompt_from_messages(path, thinking, false)` or `product::prompt_from_text`, not a local message parser. It supports text-only, new assistant turns, no tool declarations/tool history/forced tool, no media, no custom cache markers, no additional template kwargs or structured-output instruction. These modes are explicitly rejected rather than approximated. The product parser with `vision=false` rejects media before acquisition. In the admitted scope `Frontend::prepare` has no tool-boundary marker insertion, so the exact source input options and empty marker vector are replayed unchanged. The normal automatic structural cache hints remain enabled in the actual prepare call; they do not insert rendered markers in this scope.

The fixed input file set is `config.json` (at most 1 MiB), `tokenizer.json` (32 MiB), `tokenizer_config.json` (2 MiB), `generation_config.json` (1 MiB), `chat_template.jinja` (2 MiB). Prompt/messages input is at most 2 MiB. No directory scan, `.ninfer` artifact read, weight load, Engine instance, or DeviceContext is present. Vision/ngram options are disabled. `parse_resources` is the production tokenizer/contiguous-public-domain validator, using a Config with the physical `text_config.vocab_size` row bound only; this is not full model/config/vision/draft validation.

There is a real production coupling: `Frontend::prepare` calls `kvmem_publish_query_span` (`frontend.cpp:1103`), defined in `program/prefill.cpp:1459`. That function invokes `kvmem_score_set_query_span` (`ops/kvmem/kvmem_score.h:198`), which writes two integers in the default-constructed process-static probe (`:155`), with no allocation or device call on this path. Do not replace it with a no-op. The proposed first link reuses the original frontend test's full Engine library closure, so the executable **executes only CPU preparation but still links HIP and device-containing archives**. A strictly HIP-free binary would require a separate production ownership change; it is not claimed here. Static module/runtime registration at process startup must also be observed during future actual host-only qualification.

## Proposed CMake hook (root owns integration)

After the existing Engine and Products targets are available:

```cmake
add_executable(ninfer_amd_frontend_export
  "${CMAKE_CURRENT_LIST_DIR}/engine/tools/frontend_export.cpp")
ninfer_native_hip(ninfer_amd_frontend_export)
target_link_libraries(ninfer_amd_frontend_export PRIVATE
  ninfer_engine_amd ninfer_core_amd ninfer_product_prompt_input_amd ninfer::json)
ninfer_product_stage_runtime(ninfer_amd_frontend_export)
```

`ninfer_native_hip` supplies the existing generated `src`/`include` paths before the upstream paths and the actual HIP definitions/link options. The exporter itself is `.cpp`, not a HIP kernel translation unit. Its include must resolve to `${_generated}/src/models/qwen3_5/frontend/frontend.cpp`. Do not add a separately compiled `frontend.cpp` to this executable, do not use whole-archive, and do not add a second frontend implementation. The archive member for that frontend should remain unpulled because this TU already defines all its symbols; this needs real link verification. The existing Engine transitive closure supplies qwen resource/tokenizer, processor, output session, prompt layout, tool parser, ngram/cache/tap, text/utf8proc, Jinja, and the original prefill publisher. Products supplies the unchanged messages parser and its transitive media-acquire/curl system dependencies. No production test source or math criterion is altered.

## Schema version 1

Required alignment fields are:

- `schema_version: 1`, `host_only: true` (execution scope; not a HIP-free-link claim).
- `rendered_utf8`: the complete production rendered string before tokenizer NFC/BPE; `rendered_byte_count` counts UTF-8 bytes including NUL and controls.
- `literal_spans`: ordered, disjoint, nonempty `{byte_offset, byte_length}` byte ranges within that string; not character offsets. They must be preserved when encoding it elsewhere.
- `token_ids`: all actual prepared IDs, in order; `token_count` is its length. No inferred BOS/EOS or reconstructed string is substituted.
- `token_pieces`: one per prepared ID, with actual `id`, `bytes_hex` (authoritative decoded bytes), `special`, and `utf8` (string if valid, otherwise null). A token can contain a partial Unicode codepoint. Concatenated decoded bytes need not equal raw rendered bytes because the native tokenizer performs NFC.
- `public_token_count`, `physical_vocab_size`, and the half-open `excluded_physical_token_range: [public_token_count, physical_vocab_size]`. Bonsai expected values remain 248077 and 248320, to be established by the actual run.
- `special_tokens`: real `Tokenizer::is_special_token` IDs and byte views. Ordinary added tokens such as the think delimiters are not silently classified special.
- `default_stop_token_ids`, `caller_stop_token_ids`, `effective_stop_token_ids`: actual production `merge_stop_policy` output preserves default/caller order and deduplicates. Stops outside the public domain are rejected.
- `starts_in_reasoning`, `prompt_options.enable_thinking` (false/true/null), `stop_policy`, `validation`, and the explicit input scope/paths/max context.

JSON is serialized strictly. NUL, CRLF, quotes and controls are JSON escapes; valid Unicode is preserved. Invalid per-token UTF-8 is hex, never replacement characters. Root may compute SHA256 from decoded `rendered_utf8.encode('utf-8')`, serialized ID arrays, fixture/resource bytes with its Python sidecar; no C++ crypto dependency was added. Do not hash the JSON escaping and call it a rendered byte hash.

## Candidate commands after a successful build

These are new research-tool arguments, not pre-existing native CLI flags:

```powershell
$exportExe = 'D:/programfile/work/deepseek/sources/ninfer-fusion-kvmem/ports/amd/build/ninfer_amd_frontend_export.exe'
$resourceDir = 'D:/programfile/work/deepseek/artifacts/ninfer-amd/bonsai2-resources'
$fixtureDir = 'D:/programfile/work/deepseek/sources/ninfer-fusion-kvmem/ports/amd/engine/tools'
& $exportExe --resources $resourceDir --messages "$fixtureDir/frontend_export_fixture_en.json" --thinking off --max-context 8192 --output 'D:/programfile/work/deepseek/research/bonsai-frontend-en.json'
& $exportExe --resources $resourceDir --messages "$fixtureDir/frontend_export_fixture_literal_controls.json" --thinking off --max-context 8192 --output 'D:/programfile/work/deepseek/research/bonsai-frontend-literal-controls.json'
```

Use UTF-8 JSON or `--prompt-file` for Unicode input on Windows; the narrow `--prompt` argv must itself already be UTF-8. `frontend_export_fixtures.json` lists five positive fixtures, two rejection fixtures, thinking modes, stop dedup and the first padded-row rejection. It deliberately contains no fabricated expected native IDs. The English text matches the existing native smoke text, whose historical log reported 19 prompt tokens, but the new exporter still requires an actual run. A successful source/JSON check is not a native tokenizer or model PASS.

After build, check every positive fixture exits zero and produces valid schema 1 JSON; replay identity is enforced inside the executable. Check all rejection cases exit nonzero and produce no success JSON. Then compare complete rendered UTF-8 bytes, literal-span-aware actual ID vectors, public/logprob domains, and effective stop sets against independently captured actual Prism/native frontend outputs. Calling HF tokenize on rendered text does not establish this alignment. Prompt alignment alone does not qualify generation, teacher-forced quality, memory or throughput, and the user's formal-migration gate of a significant advantage over llama remains in force.
