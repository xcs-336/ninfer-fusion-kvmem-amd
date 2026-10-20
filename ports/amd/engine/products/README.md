`Products.cmake` builds the actual original `ninfer`, `ninfer-serve`,
`ninfer-calibrate`, `ninfer-perplexity` and `ninfer_bench` programs. Generation,
scoring and measurements use the same complete public `.ninfer` Engine.
Original CLI options, input acquisition, logs, OpenAI/Anthropic serving,
benchmark measurements and optional local WebUI embedding retain their owners.

The source manifest preserves six product support TUs, 27 serving TUs plus the
original WebUI source choice, nine original executable TUs, and all 19 original
product/benchmark-helper tests. It leaves the Engine source manifest untouched.
The new native environment test independently checks HIP version decoding,
actual architecture fields and omission of CUDA/compute-capability metadata.
The original synthetic CUDA formatter fixtures and assertions are unchanged.

Real native reports use HIP's `major*10^7 + minor*10^5 + patch` encoding and the
actual `hipDeviceProp_t::gcnArchName`. Calibration uses the same shared
`ninfer::amd::context_cost_hardware_class(name, gcn_arch_name)` as the Engine,
including target-ID feature flags. No NVIDIA SM identity is synthesized.
Build-time stamping runs the original Git identity script with the actual AMD
port and benchmark source roots included in its dirty check.

The configured libcurl package supplies matching headers, import library and
DLL. Its four actual non-system DLL dependencies are resolved recursively from
PE import tables, recorded with hashes and staged alongside the programs;
FFmpeg uses the already configured complete dependency package. ROCm remains
in its real installed layout, including `.kpack` and rocBLAS kernel data.
`ninfer-amd-env.cmd` only prepends that SDK's `bin` directory to `PATH` before
executing a supplied original program. From the build directory, for example:

```powershell
.\ninfer-amd-env.cmd .\ninfer.exe --help
.\ninfer-amd-env.cmd .\ninfer-perplexity.exe --help
.\ninfer-amd-env.cmd .\ninfer_bench.exe --help
```

The actual CLI, benchmark and perplexity tools expose the original Engine's
`--device-profile auto|off|calibrate`. The default remains `auto`, which measures
real routes when no actual hardware profile exists. `off` disables lookup and
calibration for a bounded first run; `calibrate` requests real measurements.
The mode is recorded in CLI logs, benchmark table/JSON/CSV and perplexity reports.
The existing CLI/benchmark parser tests retain their assertions and add valid,
invalid, missing-value and default cases. The additional CPU-only
`ninfer_amd_device_profile_cli_contract` invokes the actual three programs'
help/error paths before Engine or GPU initialization, including perplexity's
otherwise private parser. The real serving tool already exposes its original
profile mode and path options.

Source generation, dependency resolution, script syntax, in-place source-view
idempotence, the actual build-id script and the initial metadata/benchmark/CLI
frontends pass. Full product compilation/linking, the original product tests,
the native format oracle and actual `.ninfer` generation/scoring/benchmarks
remain to be run through the shared build and execution schedule. No
end-to-end performance result is established by these frontend checks.
