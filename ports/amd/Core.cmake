set(_engine "${CMAKE_CURRENT_LIST_DIR}/../../src-tree/fusion-engine-src")
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_prepared "${CMAKE_CURRENT_BINARY_DIR}/generated-stage")
file(GLOB_RECURSE _core_inputs CONFIGURE_DEPENDS
  "${_engine}/src/*.h" "${_engine}/src/*.hpp" "${_engine}/src/*.cuh" "${_engine}/src/*.cpp" "${_engine}/src/*.cu"
  "${_engine}/include/*.h" "${_engine}/include/*.hpp" "${_engine}/tests/*.h" "${_engine}/tests/*.hpp" "${_engine}/tests/*.cuh"
  "${_engine}/tests/*.cpp" "${_engine}/tests/*.cu"
  "${_engine}/apps/*.h" "${_engine}/apps/*.hpp" "${_engine}/apps/*.cuh"
  "${_engine}/apps/*.cpp" "${_engine}/apps/*.cu"
  "${_engine}/bench/*.h" "${_engine}/bench/*.hpp" "${_engine}/bench/*.cuh"
  "${_engine}/bench/*.cpp" "${_engine}/bench/*.cu")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_core_inputs}
  "${CMAKE_CURRENT_LIST_DIR}/prepare_core.py" "${CMAKE_CURRENT_LIST_DIR}/core/hip_vmm_address.h"
  "${CMAKE_CURRENT_LIST_DIR}/core/hip_graph_instantiate.h"
  "${CMAKE_CURRENT_LIST_DIR}/core/hip_profiler.cpp" "${CMAKE_CURRENT_LIST_DIR}/core/hip_intrinsics.h"
  "${CMAKE_CURRENT_LIST_DIR}/core/pdl.cuh" "${CMAKE_CURRENT_LIST_DIR}/ops/memory.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/ops/ldmatrix.cuh" "${CMAKE_CURRENT_LIST_DIR}/ops/mma.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/ops/mma_f16_gfx12.cuh" "${CMAKE_CURRENT_LIST_DIR}/ops/mma_s8_gfx12.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/ops/mma_fp8_gfx12.cuh" "${CMAKE_CURRENT_LIST_DIR}/ops/mma_nvfp4_gfx12.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/gfx12-mma/mma_bf16_gfx12.h"
  "${CMAKE_CURRENT_LIST_DIR}/sync_source_view.py")
execute_process(COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/prepare_core.py"
  --sdk "${ROCM_ROOT}" --output "${_prepared}"
  RESULT_VARIABLE _generation_status COMMAND_ECHO STDOUT)
if(NOT _generation_status EQUAL 0)
  message(FATAL_ERROR "HIP Core source generation failed")
endif()
file(READ "${_engine}/src/core/CMakeLists.txt" _core_build)
string(REGEX MATCHALL "\n[ \t]+[A-Za-z_]+\\.(cu|cpp)" _source_names "${_core_build}")
set(_core_sources)
foreach(_name IN LISTS _source_names)
  string(STRIP "${_name}" _name)
  list(APPEND _core_sources "${_generated}/src/core/${_name}")
endforeach()
list(APPEND _core_sources "${_generated}/src/core/hip_profiler.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/Memory.cmake")
function(ninfer_native_hip target)
  target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__=1 NINFER_HIP=1 NOMINMAX WIN32_LEAN_AND_MEAN)
  target_include_directories(${target} PUBLIC "${ROCM_ROOT}/include" "${_generated}/src" "${_generated}/include" "${_engine}/src" "${_engine}/include")
  target_link_libraries(${target} PUBLIC "${ROCM_ROOT}/lib/amdhip64.lib")
  target_link_options(${target} PUBLIC --hip-link --offload-arch=gfx1201 "--rocm-path=${ROCM_ROOT}")
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_source_files_properties("${_source}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS
        "-x;hip;--offload-arch=gfx1201;-mno-wavefrontsize64;-include;${_generated}/src/core/hip_intrinsics.h;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    endif()
  endforeach()
endfunction()
add_library(ninfer_core_amd STATIC ${_core_sources})
ninfer_native_hip(ninfer_core_amd)
find_package(Threads REQUIRED)
target_link_libraries(ninfer_core_amd PUBLIC Threads::Threads advapi32)
foreach(_test decode_graph arena arena_ranks vmm_graph_remap evictable_kv_pool
  evictable_weight_pool tensor layout disk_kv_store disk_kv_bridge kv_cache
  kv_cache_ranks kv_capacity state_store)
  if(EXISTS "${_engine}/tests/test_${_test}.cu")
    set(_source "${_generated}/tests/test_${_test}.cu")
  else()
    set(_source "${_generated}/tests/test_${_test}.cpp")
  endif()
  set(_target "ninfer_amd_${_test}_test")
  add_executable(${_target} "${_source}")
  if(_test STREQUAL "kv_cache_ranks")
    file(WRITE "${_prepared}/tests/kv_cache_ranks_main.cpp"
      "int run_kv_cache_ranks_test();\nint main() { return run_kv_cache_ranks_test(); }\n")
    target_sources(${_target} PRIVATE "${_generated}/tests/kv_cache_ranks_main.cpp")
  elseif(_test STREQUAL "kv_capacity")
    target_sources(${_target} PRIVATE "${_engine}/src/runtime/engine/kv_capacity.cpp")
  endif()
  ninfer_native_hip(${_target})
  target_link_libraries(${_target} PRIVATE ninfer_core_amd)
  target_include_directories(${_target} PRIVATE "${_generated}/tests" "${_engine}/tests")
  ninfer_add_gpu_test(${_target} ${_target} 30)
endforeach()
ninfer_amd_memory_tests()
option(NINFER_AMD_BASIC_OPS "Build production basic operators and their original numerical tests" OFF)
add_executable(ninfer_amd_production_bf16_gemm_test tests/gemm_bf16.cu)
ninfer_native_hip(ninfer_amd_production_bf16_gemm_test)
target_include_directories(ninfer_amd_production_bf16_gemm_test PRIVATE "${_generated}/tests")
target_link_libraries(ninfer_amd_production_bf16_gemm_test PRIVATE ninfer_core_amd)
target_compile_options(ninfer_amd_production_bf16_gemm_test PRIVATE -fno-fast-math -ffp-contract=off)
ninfer_add_gpu_test(native_production_bf16_gemm ninfer_amd_production_bf16_gemm_test 60)
add_executable(ninfer_amd_matrix_f32_test tests/matrix_f32.cu)
ninfer_native_hip(ninfer_amd_matrix_f32_test)
target_include_directories(ninfer_amd_matrix_f32_test PRIVATE "${_generated}/tests")
target_link_libraries(ninfer_amd_matrix_f32_test PRIVATE ninfer_core_amd)
target_compile_options(ninfer_amd_matrix_f32_test PRIVATE -fno-fast-math -ffp-contract=off)
ninfer_add_gpu_test(native_matrix_f32 ninfer_amd_matrix_f32_test 30)
add_executable(ninfer_amd_matrix_s8_test tests/matrix_s8.cu)
ninfer_native_hip(ninfer_amd_matrix_s8_test)
target_include_directories(ninfer_amd_matrix_s8_test PRIVATE "${_generated}/tests")
target_link_libraries(ninfer_amd_matrix_s8_test PRIVATE ninfer_core_amd)
target_compile_options(ninfer_amd_matrix_s8_test PRIVATE -fno-fast-math -ffp-contract=off)
ninfer_add_gpu_test(native_matrix_s8 ninfer_amd_matrix_s8_test 30)
if(NINFER_AMD_BASIC_OPS)
  include("${CMAKE_CURRENT_LIST_DIR}/BasicOps.cmake")
endif()
option(NINFER_AMD_ARTIFACT "Build the real v3 artifact loader/materializer and original tests" OFF)
if(NINFER_AMD_ARTIFACT)
  include("${CMAKE_CURRENT_LIST_DIR}/Artifact.cmake")
endif()
option(NINFER_AMD_RUNTIME_SUPPORT "Build real Engine admission/context/profile support and original tests" OFF)
if(NINFER_AMD_RUNTIME_SUPPORT)
  include("${CMAKE_CURRENT_LIST_DIR}/RuntimeSupport.cmake")
endif()
option(NINFER_AMD_ENGINE "Build the original public Engine and complete fixed model implementations" OFF)
if(NINFER_AMD_ENGINE)
  include("${CMAKE_CURRENT_LIST_DIR}/Engine.cmake")
endif()
option(NINFER_AMD_PRODUCTS "Build the original CLI, server, calibration, perplexity and inference benchmark" OFF)
if(NINFER_AMD_PRODUCTS)
  include("${CMAKE_CURRENT_LIST_DIR}/Products.cmake")
endif()
option(NINFER_AMD_ALIGNMENT_TOOLS "Build optional native prompt/token scoring research tools" OFF)
if(NINFER_AMD_ALIGNMENT_TOOLS)
  if(NOT NINFER_AMD_PRODUCTS)
    message(FATAL_ERROR "Alignment tools require the actual Engine and product prompt-input dependencies")
  endif()
  include("${CMAKE_CURRENT_LIST_DIR}/engine/AlignmentTools.cmake")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/OriginalOps.cmake")
# Publish the fully adapted source view once. Configuring an unchanged family
# must not change timestamps and trigger expensive device-code recompilation.
execute_process(COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/sync_source_view.py"
  --source "${_prepared}" --destination "${_generated}"
  RESULT_VARIABLE _sync_status COMMAND_ECHO STDOUT)
if(NOT _sync_status EQUAL 0)
  message(FATAL_ERROR "HIP source view publication failed")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/memory/SegmentedStorage.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/memory/SegmentedPool.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/memory/SegmentedBF16Ops.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/memory/readonly_table/SegmentedBF16Readonly.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/memory/b1_public/PublicBF16.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/memory/controller_handoff/ControllerHandoff.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/memory/models_storage/ModelsStorage.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/memory/models_engine_ops/ModelsEngineOps.cmake")
# The full ordinary cache and copied B1 dispatcher have separate route helpers.
# Preserve both definitions and their bodies with target-private symbol names.
if(TARGET ninfer_models_engine_b1_ops_amd)
  target_compile_definitions(ninfer_models_engine_b1_ops_amd PRIVATE
    small_t_i8_route_key=ninfer_amd_models_b1_small_t_i8_route_key
    small_t_i8_routed_ctas_per_sm=ninfer_amd_models_b1_small_t_i8_routed_ctas_per_sm)
endif()
# The opt-in fixture includes the parent Core using its amd-memory/ prefix.
# Keep this usage requirement local to the fixture; library source recipes stay frozen.
if(TARGET ninfer_amd_models_engine_b1_ops_test)
  target_include_directories(ninfer_amd_models_engine_b1_ops_test PRIVATE
    "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/memory/models_engine_worker/ModelsEngineWorker.cmake")
# Explicit default-OFF full-Core capacity test; no production gate or automatic run.
include("${CMAKE_CURRENT_LIST_DIR}/memory/vision_segmented_capacity_test/VisionSegmentedCapacityTest.cmake")
# Explicit default-OFF real Broker component and physical fixture.
include("${CMAKE_CURRENT_LIST_DIR}/memory/segmented_vision_broker/SegmentedVisionBroker.cmake")
# Experimental leaves share source inputs with ordinary Ops. CMake retains
# lexical parent-directory aliases, whereas Ninja canonicalizes output paths.
# Keep the complete regeneration dependency set with one absolute spelling.
get_property(_amd_configure_dependencies DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
set(_amd_normalized_configure_dependencies)
foreach(_amd_dependency IN LISTS _amd_configure_dependencies)
  get_filename_component(_amd_absolute_dependency "${_amd_dependency}" ABSOLUTE)
  list(APPEND _amd_normalized_configure_dependencies "${_amd_absolute_dependency}")
endforeach()
list(REMOVE_DUPLICATES _amd_normalized_configure_dependencies)
set_property(DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS ${_amd_normalized_configure_dependencies})
