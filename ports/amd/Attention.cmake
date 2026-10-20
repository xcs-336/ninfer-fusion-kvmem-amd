# Build complete production storage/attention families, rather than a BF16-only
# subset whose missing symbols would surface only at full Engine link time.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/attention/prepare_attention.py"
  "${CMAKE_CURRENT_LIST_DIR}/attention/partial_barrier.cuh")
execute_process(COMMAND "${NINFER_PYTHON}"
  "${CMAKE_CURRENT_LIST_DIR}/attention/prepare_attention.py" --generated "${_prepared}"
  RESULT_VARIABLE _attention_generation_status COMMAND_ECHO STDOUT)
if(NOT _attention_generation_status EQUAL 0)
  message(FATAL_ERROR "Native HIP attention source preparation failed")
endif()

set(_attention_paths)
set(_attention_storage_paths)
set(_attention_context_paths)
set(_attention_packed_paths)
foreach(_family softmax_attention softmax_attention/selected_block kv_cache context_kv_materialize kv_stream)
  file(READ "${_engine}/src/ops/${_family}/sources.cmake" _attention_manifest)
  string(REGEX MATCHALL "[A-Za-z_0-9/]+\\.(cu|cpp)" _attention_sources "${_attention_manifest}")
  foreach(_source IN LISTS _attention_sources)
    string(REGEX REPLACE "^/+" "" _source "${_source}")
    set(_path "${_generated}/src/ops/${_family}/${_source}")
    if(NOT _family STREQUAL "softmax_attention")
      list(APPEND _attention_storage_paths "${_path}")
    elseif(_source MATCHES "^dense/context/|^sliding_window/")
      list(APPEND _attention_context_paths "${_path}")
    elseif(_source MATCHES "^dense/packed/")
      list(APPEND _attention_packed_paths "${_path}")
    else()
      list(APPEND _attention_paths "${_path}")
    endif()
  endforeach()
endforeach()
# Independent original Op tests can link their closed family while causal
# attention compiles; the complete Engine-facing target still requires every
# production source and storage format through these explicit dependencies.
foreach(_part storage context packed)
  add_library(ninfer_attention_${_part}_ops_amd STATIC ${_attention_${_part}_paths})
  ninfer_native_hip(ninfer_attention_${_part}_ops_amd)
  target_link_libraries(ninfer_attention_${_part}_ops_amd PUBLIC ninfer_basic_ops_amd)
endforeach()
add_library(ninfer_attention_ops_amd STATIC ${_attention_paths})
ninfer_native_hip(ninfer_attention_ops_amd)
target_link_libraries(ninfer_attention_ops_amd PUBLIC ninfer_basic_ops_amd
  ninfer_attention_storage_ops_amd ninfer_attention_context_ops_amd ninfer_attention_packed_ops_amd)

# Keep the unchanged upstream FP64 attention oracle and all its format modes.
set(_attention_test ninfer_amd_softmax_attention_op_test)
add_executable(${_attention_test}
  "${_generated}/tests/ops/softmax_attention/main.cpp"
  "${_generated}/tests/ops/softmax_attention/causal_cache.cpp"
  "${_generated}/tests/ops/softmax_attention/plain_and_packed.cpp"
  "${_generated}/tests/ops/softmax_attention/context.cpp")
ninfer_native_hip(${_attention_test})
target_include_directories(${_attention_test} PRIVATE "${_generated}/tests")
target_compile_options(${_attention_test} PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(${_attention_test} PRIVATE ninfer_attention_ops_amd)
ninfer_add_gpu_test(${_attention_test} ${_attention_test} 1500)
foreach(_mode nvfp4 k8v4 rk4v4-e8 rk2v4-e8 int8-prompt pack-gqa dflash2 wide)
  ninfer_add_gpu_test("ninfer_amd_attention_${_mode}" ${_attention_test} 1500 "--${_mode}-only")
endforeach()
foreach(_test sliding_window_attention context_kv_materialize kv_cache_append kv_plane_types kv_stream)
  set(_target "ninfer_amd_${_test}_op_test")
  add_executable(${_target} "${_generated}/tests/ops/test_${_test}.cpp")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  if(_test STREQUAL "sliding_window_attention")
    target_link_libraries(${_target} PRIVATE ninfer_attention_context_ops_amd)
  else()
    target_link_libraries(${_target} PRIVATE ninfer_attention_storage_ops_amd)
  endif()
  if(_test STREQUAL "kv_plane_types")
    # This executable contains only static_asserts and a host printf.
    add_test(NAME ${_target} COMMAND "$<TARGET_FILE:${_target}>")
    set_tests_properties(${_target} PROPERTIES TIMEOUT 30)
  else()
    ninfer_add_gpu_test(${_target} ${_target} 300)
  endif()
endforeach()
# These entry wrappers invoke the unchanged upstream suites from the combined
# test directly. Their independent FP64 oracle and complete case domains stay
# in the original sources; no test-owned GPU arithmetic is introduced.
foreach(_part context plain_and_packed)
  set(_target "ninfer_amd_${_part}_attention_op_test")
  file(WRITE "${_prepared}/tests/ops/softmax_attention/${_part}_main.cpp"
    "#include <cstdio>\n#include <exception>\nint run_softmax_attention_${_part}_tests();\nint main() { try { return run_softmax_attention_${_part}_tests(); } catch (const std::exception& error) { std::fprintf(stderr, \"${_part} attention: %s\\n\", error.what()); return 1; } }\n")
  add_executable(${_target} "${_generated}/tests/ops/softmax_attention/${_part}.cpp"
    "${_generated}/tests/ops/softmax_attention/${_part}_main.cpp")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  if(_part STREQUAL "context")
    target_link_libraries(${_target} PRIVATE ninfer_attention_context_ops_amd)
  else()
    target_link_libraries(${_target} PRIVATE ninfer_attention_packed_ops_amd)
  endif()
  ninfer_add_gpu_test(${_target} ${_target} 1500)
endforeach()
foreach(_mode nvfp4 k8v4)
  ninfer_add_gpu_test("ninfer_amd_kv_cache_append_${_mode}"
    ninfer_amd_kv_cache_append_op_test 300 "--${_mode}-only")
endforeach()

# The upstream KVstream canary protects transfers and decode state. Its public
# shared-row WMMA prefill has no full mathematical oracle there, so qualify
# that production Op directly after each represented BF16/FP8 input boundary.
add_executable(ninfer_amd_selected_block_attention_op_test attention/test_selected_block.cpp)
ninfer_native_hip(ninfer_amd_selected_block_attention_op_test)
target_include_directories(ninfer_amd_selected_block_attention_op_test PRIVATE "${_generated}/tests")
target_compile_options(ninfer_amd_selected_block_attention_op_test PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(ninfer_amd_selected_block_attention_op_test PRIVATE ninfer_attention_storage_ops_amd)
ninfer_add_gpu_test(ninfer_amd_selected_block_attention_op_test
  ninfer_amd_selected_block_attention_op_test 60)

# This narrow exact test qualifies the replacement for PTX's producer-only
# named barriers independently before the enclosing arithmetic is qualified.
add_executable(ninfer_amd_attention_partial_barrier_test attention/test_partial_barrier.cu)
ninfer_native_hip(ninfer_amd_attention_partial_barrier_test)
target_include_directories(ninfer_amd_attention_partial_barrier_test PRIVATE "${CMAKE_CURRENT_LIST_DIR}/attention")
ninfer_add_gpu_test(native_attention_partial_barrier
  ninfer_amd_attention_partial_barrier_test 30)
