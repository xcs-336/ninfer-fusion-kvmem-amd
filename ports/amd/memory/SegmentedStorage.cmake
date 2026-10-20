# Isolated experimental tests only. Default OFF; this does not replace the
# original DeviceKVPagePool or change its VMM/Attention/Engine admission.
option(NINFER_AMD_SEGMENTED_STORAGE_TESTS "Build isolated segmented KV planner/Core byte tests" OFF)
if(NOT NINFER_AMD_SEGMENTED_STORAGE_TESTS)
  return()
endif()
if(NOT COMMAND ninfer_native_hip OR NOT COMMAND ninfer_add_gpu_test)
  message(FATAL_ERROR "Include SegmentedStorage.cmake after the native HIP/test helpers")
endif()
if(NOT WIN32)
  message(FATAL_ERROR "This experimental checkpoint is qualified only on native Windows gfx1201")
endif()

set(_segmented_storage_dir "${CMAKE_CURRENT_LIST_DIR}")

# Pure host C++: intentionally no ninfer_native_hip, HIP definitions, runtime,
# force-included device helpers, production archive or device compilation.
add_executable(ninfer_amd_kv_segmented_plan_test
  "${_segmented_storage_dir}/tests/kv_segmented_plan_test.cpp"
  "${_segmented_storage_dir}/kv_segmented_loan_plan.cpp")
target_include_directories(ninfer_amd_kv_segmented_plan_test PRIVATE "${_segmented_storage_dir}")
target_compile_features(ninfer_amd_kv_segmented_plan_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_kv_segmented_plan_test PRIVATE
  -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
add_test(NAME native_kv_segmented_plan COMMAND $<TARGET_FILE:ninfer_amd_kv_segmented_plan_test>)
set_tests_properties(native_kv_segmented_plan PROPERTIES TIMEOUT 120 LABELS "amd;experimental;host")

# Actual small HIP TU plus its own host transport/planner objects. Reuse the
# real gfx1201/wave32 SDK helper; do not link ninfer_core_amd or an original pool.
add_executable(ninfer_amd_kv_segmented_storage_test
  "${_segmented_storage_dir}/tests/kv_segmented_storage_test.cu"
  "${_segmented_storage_dir}/kv_segmented_storage.cpp"
  "${_segmented_storage_dir}/kv_segmented_loan_plan.cpp")
ninfer_native_hip(ninfer_amd_kv_segmented_storage_test)
target_include_directories(ninfer_amd_kv_segmented_storage_test PRIVATE "${_segmented_storage_dir}")
target_compile_features(ninfer_amd_kv_segmented_storage_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_kv_segmented_storage_test PRIVATE
  -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
target_link_options(ninfer_amd_kv_segmented_storage_test PRIVATE
  "-Wl,/MAP:${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_kv_segmented_storage_test.map")
ninfer_add_gpu_test(native_kv_segmented_storage ninfer_amd_kv_segmented_storage_test 120 --run)
set_tests_properties(native_kv_segmented_storage PROPERTIES
  LABELS "amd;experimental;functional;gpu"
  ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")

# Separate controlled-pending fixture. Its source matches the independently
# passed direct-command v2 test; this new CMake target still needs validation.
add_executable(ninfer_amd_kv_segmented_storage_pending_test
  "${_segmented_storage_dir}/tests/kv_segmented_storage_pending_test.cu"
  "${_segmented_storage_dir}/kv_segmented_storage.cpp"
  "${_segmented_storage_dir}/kv_segmented_loan_plan.cpp")
ninfer_native_hip(ninfer_amd_kv_segmented_storage_pending_test)
target_include_directories(ninfer_amd_kv_segmented_storage_pending_test PRIVATE "${_segmented_storage_dir}")
target_compile_features(ninfer_amd_kv_segmented_storage_pending_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_kv_segmented_storage_pending_test PRIVATE
  -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
target_link_options(ninfer_amd_kv_segmented_storage_pending_test PRIVATE
  "-Wl,/MAP:${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_kv_segmented_storage_pending_test.map")
ninfer_add_gpu_test(native_kv_segmented_storage_pending ninfer_amd_kv_segmented_storage_pending_test 120 --run)
set_tests_properties(native_kv_segmented_storage_pending PROPERTIES
  LABELS "amd;experimental;controlled-pending;gpu"
  ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")
# The standard GPU helper owns byte-0/process cleanup. Resource-qualified runs
# still require the owner's external host guard and actual-PID monitoring.
