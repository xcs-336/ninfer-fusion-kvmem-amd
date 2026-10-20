# The actual scheduler/resource/profile support used by the public Engine.
# This component does not claim that the model execution closure is complete.
if(NOT TARGET ninfer_basic_ops_amd)
  message(FATAL_ERROR "NINFER_AMD_RUNTIME_SUPPORT requires NINFER_AMD_BASIC_OPS")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/runtime/prepare_runtime.py"
  "${CMAKE_CURRENT_LIST_DIR}/runtime/amd_hardware_class.h")
execute_process(COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/runtime/prepare_runtime.py"
  --source-view "${_prepared}" --output "${_prepared}"
  RESULT_VARIABLE _runtime_preparation_status COMMAND_ECHO STDOUT)
if(NOT _runtime_preparation_status EQUAL 0)
  message(FATAL_ERROR "Native AMD runtime hardware identity preparation failed")
endif()
add_executable(ninfer_amd_hardware_class_test "${CMAKE_CURRENT_LIST_DIR}/runtime/test_hardware_class.cpp")
target_compile_features(ninfer_amd_hardware_class_test PRIVATE cxx_std_20)
add_test(NAME ninfer_amd_hardware_class_test COMMAND ninfer_amd_hardware_class_test)
set_tests_properties(ninfer_amd_hardware_class_test PROPERTIES TIMEOUT 10)
set(NINFER_DEVICE_PROFILES_FILE "${_engine}/src/runtime/engine/device_profiles.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NINFER_DEVICE_PROFILES_FILE}")
file(READ "${NINFER_DEVICE_PROFILES_FILE}" NINFER_DEVICE_PROFILES_JSON)
configure_file("${_engine}/src/runtime/engine/device_profiles_builtin.cpp.in"
  "${_prepared}/src/runtime/engine/device_profiles_builtin.cpp" @ONLY)

set(_runtime_support_sources
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/device_profile.cpp
  engine/device_profiles_builtin.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
  prefix_cache/block_hash.cpp
  prefix_cache/prefix_index.cpp
  prefix_cache/tap_planner.cpp)
list(TRANSFORM _runtime_support_sources PREPEND "${_generated}/src/runtime/")
add_library(ninfer_runtime_support_amd STATIC ${_runtime_support_sources})
ninfer_native_hip(ninfer_runtime_support_amd)
target_include_directories(ninfer_runtime_support_amd PUBLIC "${_engine}/third_party")
target_link_libraries(ninfer_runtime_support_amd PUBLIC ninfer_core_amd ninfer_basic_ops_amd)

# These original tests exercise CPU admission, page ownership, context budgeting,
# prefix identity and device-profile serialization. Engine/model tests remain
# with their actual owner until the complete Engine is linked.
foreach(_test admission_policy context_cost resource_manager device_profile prefix_cache_index)
  set(_target "ninfer_amd_${_test}_test")
  add_executable(${_target} "${_generated}/tests/test_${_test}.cpp")
  ninfer_native_hip(${_target})
  target_link_libraries(${_target} PRIVATE ninfer_runtime_support_amd)
  if(_test STREQUAL "device_profile")
    # Profile install/query calls HIP's current-device API even though the
    # profile serialization/resource assertions themselves run on the CPU.
    ninfer_add_gpu_test(${_target} ${_target} 30)
  else()
    add_test(NAME ${_target} COMMAND ${_target})
    set_tests_properties(${_target} PROPERTIES TIMEOUT 60)
  endif()
endforeach()
