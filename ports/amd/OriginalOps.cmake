# Remaining original programs require families configured after BasicOps.
# Keep their real sources, numerical oracles and supported-domain checks.
if(TARGET ninfer_attention_storage_ops_amd)
  add_executable(ninfer_amd_e8_root_decode_op_test
    "${_generated}/tests/ops/test_e8_root_decode.cu")
  ninfer_native_hip(ninfer_amd_e8_root_decode_op_test)
  target_include_directories(ninfer_amd_e8_root_decode_op_test PRIVATE "${_generated}/tests")
  target_compile_options(ninfer_amd_e8_root_decode_op_test PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(ninfer_amd_e8_root_decode_op_test PRIVATE ninfer_attention_storage_ops_amd)
  ninfer_add_gpu_test(ninfer_amd_e8_root_decode_op_test ninfer_amd_e8_root_decode_op_test 1500)
endif()

if(TARGET ninfer_artifact_amd AND TARGET ninfer_linear_ops_amd)
  add_executable(ninfer_amd_vocabulary_transcode_op_test
    "${_generated}/tests/ops/test_vocabulary_transcode.cpp")
  ninfer_native_hip(ninfer_amd_vocabulary_transcode_op_test)
  target_include_directories(ninfer_amd_vocabulary_transcode_op_test PRIVATE "${_generated}/tests")
  target_compile_options(ninfer_amd_vocabulary_transcode_op_test PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(ninfer_amd_vocabulary_transcode_op_test PRIVATE
    ninfer_linear_ops_amd ninfer_artifact_amd)
  ninfer_add_gpu_test(ninfer_amd_vocabulary_transcode_op_test ninfer_amd_vocabulary_transcode_op_test 1500)
endif()

if(TARGET ninfer_attn_input_proj_ops_amd AND TARGET ninfer_linear_add_ops_amd AND
   TARGET ninfer_linear_pair_ops_amd AND TARGET ninfer_linear_swiglu_ops_amd)
  add_executable(ninfer_amd_route_coverage_op_test
    "${_generated}/tests/ops/test_route_coverage.cpp")
  ninfer_native_hip(ninfer_amd_route_coverage_op_test)
  target_include_directories(ninfer_amd_route_coverage_op_test PRIVATE "${_generated}/tests")
  target_compile_options(ninfer_amd_route_coverage_op_test PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(ninfer_amd_route_coverage_op_test PRIVATE
    ninfer_attn_input_proj_ops_amd ninfer_linear_add_ops_amd
    ninfer_linear_pair_ops_amd ninfer_linear_swiglu_ops_amd)
  add_test(NAME ninfer_amd_route_coverage_op_test COMMAND ninfer_amd_route_coverage_op_test)
  set_tests_properties(ninfer_amd_route_coverage_op_test PROPERTIES TIMEOUT 1500 LABELS "cpu;route_coverage")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/OriginalModes.cmake")
