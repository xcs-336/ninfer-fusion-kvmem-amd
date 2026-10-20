# Native weight borrowing uses the actual hipMalloc suffix rather than Windows
# VMM aliases. The original byte-exact behavioral test remains unchanged.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/memory/evictable_weight_pool.h"
  "${CMAKE_CURRENT_LIST_DIR}/memory/evictable_weight_pool.cpp")
configure_file("${CMAKE_CURRENT_LIST_DIR}/memory/evictable_weight_pool.h"
  "${_prepared}/src/core/evictable_weight_pool.h" COPYONLY)
list(FILTER _core_sources EXCLUDE REGEX "/evictable_weight_pool\\.cu$")
list(APPEND _core_sources "${CMAKE_CURRENT_LIST_DIR}/memory/evictable_weight_pool.cpp")

function(ninfer_amd_memory_tests)
  add_executable(ninfer_amd_memory_weight_graph_test
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tests/memory_weight_graph.cu")
  ninfer_native_hip(ninfer_amd_memory_weight_graph_test)
  target_link_libraries(ninfer_amd_memory_weight_graph_test PRIVATE ninfer_core_amd)
  ninfer_add_gpu_test(native_memory_weight_graph ninfer_amd_memory_weight_graph_test 30)
endfunction()
