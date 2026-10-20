# Keep the original closed GDN family, including both prefill algorithms.
file(READ "${_engine}/src/ops/linear_attention/sources.cmake" _gdn_manifest)
string(REGEX MATCHALL "[A-Za-z_0-9/]+\\.(cu|cpp)" _gdn_sources "${_gdn_manifest}")
set(_gdn_paths)
foreach(_source IN LISTS _gdn_sources)
  list(APPEND _gdn_paths "${_generated}/src/ops/linear_attention/${_source}")
endforeach()
add_library(ninfer_gdn_ops_amd STATIC ${_gdn_paths})
ninfer_native_hip(ninfer_gdn_ops_amd)
target_link_libraries(ninfer_gdn_ops_amd PUBLIC ninfer_basic_ops_amd)
# test_gdn_replay_fold also calls fused input projections, whose closed family
# is not ported yet. Retain it upstream and qualify it with that dependency.
foreach(_test gated_delta_net gated_delta_net_two_stage gdn_state_fp16
              gated_delta_net_replay_record)
  set(_target "ninfer_amd_${_test}_op_test")
  add_executable(${_target} "${_generated}/tests/ops/test_${_test}.cpp")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(${_target} PRIVATE ninfer_gdn_ops_amd)
  ninfer_add_gpu_test(${_target} ${_target} 60)
endforeach()
