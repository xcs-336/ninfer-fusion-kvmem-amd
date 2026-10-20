file(READ "${_engine}/src/ops/basic_sources.cmake" _basic_manifest)
string(REGEX MATCHALL "[A-Za-z_0-9/]+\\.(cu|cpp)" _basic_sources "${_basic_manifest}")
set(_basic_paths)
foreach(_source IN LISTS _basic_sources)
  list(APPEND _basic_paths "${_generated}/src/ops/${_source}")
endforeach()
add_library(ninfer_basic_ops_amd STATIC ${_basic_paths})
ninfer_native_hip(ninfer_basic_ops_amd)
target_link_libraries(ninfer_basic_ops_amd PUBLIC ninfer_core_amd)
include("${CMAKE_CURRENT_LIST_DIR}/Gguf.cmake")
foreach(_test rmsnorm gated_rmsnorm residual_add gelu silu_mul embedding hadamard_transform
  add_bias argmax cast causal_conv1d_silu gdn_gating l2norm layer_norm mtp_pack mtp_round
  speculative_round paged_kv_window position prepare_ragged_prefix prepare_masked_block
  rmsnorm_pack_tail rope sampling scalar scatter scatter_bf16_batch sigmoid_mul target_logprobs
  vision_pos_embed)
  set(_target "ninfer_amd_${_test}_op_test")
  add_executable(${_target} "${_generated}/tests/ops/test_${_test}.cpp")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_link_libraries(${_target} PRIVATE ninfer_basic_ops_amd)
  # Match the upstream independent host-oracle compile contract.
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  ninfer_add_gpu_test(${_target} ${_target} 60)
endforeach()
ninfer_add_gpu_test(ninfer_amd_speculative_transforms
  ninfer_amd_speculative_round_op_test 30 --transforms-only)
