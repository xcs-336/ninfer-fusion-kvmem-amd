# The original public .ninfer Engine, fixed model implementations, and frontend
# components. No alternate model loader or inference entry point is provided.
set(_engine_required_ops ninfer_basic_ops_amd ninfer_gdn_ops_amd ninfer_attention_ops_amd
  ninfer_linear_ops_amd ninfer_linear_add_ops_amd ninfer_linear_swiglu_ops_amd
  ninfer_gdn_input_proj_ops_amd ninfer_gdn_gating_proj_ops_amd ninfer_attn_input_proj_ops_amd
  ninfer_linear_pair_ops_amd ninfer_linear_topk_ops_amd ninfer_rmsnorm_rope_ops_amd
  ninfer_candidate_selector_ops_amd ninfer_dynamic_grouped_conv_ops_amd
  ninfer_kvmem_ops_amd ninfer_sparse_moe_ops_amd)
foreach(_dependency IN LISTS _engine_required_ops)
  if(NOT TARGET ${_dependency})
    message(FATAL_ERROR "Public native Engine requires the actual ${_dependency} production closure")
  endif()
endforeach()
foreach(_dependency ninfer_artifact_amd ninfer_runtime_support_amd)
  if(NOT TARGET ${_dependency})
    message(FATAL_ERROR "Public native Engine requires ${_dependency}")
  endif()
endforeach()

file(GLOB _engine_port_inputs CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/engine/*.py"
  "${CMAKE_CURRENT_LIST_DIR}/engine/*.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_engine_port_inputs})
execute_process(COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/engine/prepare_engine.py"
  --engine "${_engine}" --source-view "${_prepared}" --output "${_prepared}"
  RESULT_VARIABLE _engine_preparation_status COMMAND_ECHO STDOUT)
if(NOT _engine_preparation_status EQUAL 0)
  message(FATAL_ERROR "Public native Engine manifest/source preparation failed")
endif()
file(READ "${_prepared}/engine-manifest.json" _engine_manifest)
string(JSON _engine_input_count LENGTH "${_engine_manifest}" inputs)
math(EXPR _engine_last_input "${_engine_input_count} - 1")
foreach(_index RANGE 0 ${_engine_last_input})
  string(JSON _input GET "${_engine_manifest}" inputs ${_index})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_engine}/${_input}")
endforeach()

function(ninfer_engine_source_paths component output)
  string(JSON _count LENGTH "${_engine_manifest}" components ${component})
  math(EXPR _last "${_count} - 1")
  set(_paths)
  foreach(_index RANGE 0 ${_last})
    string(JSON _source GET "${_engine_manifest}" components ${component} ${_index})
    if(_source MATCHES "^third_party/")
      list(APPEND _paths "${_engine}/${_source}")
    else()
      list(APPEND _paths "${_generated}/${_source}")
    endif()
  endforeach()
  set(${output} "${_paths}" PARENT_SCOPE)
endfunction()

find_package(Threads REQUIRED)
# utf8proc remains C. Use the SDK's matching C frontend and CRT conventions,
# rather than compiling a third-party C implementation as C++.
if(NOT CMAKE_C_COMPILER_LOADED)
  if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${ROCM_ROOT}/lib/llvm/bin/clang.exe" CACHE FILEPATH
      "AMD Clang C compiler for original utf8proc")
  endif()
  enable_language(C)
endif()
if(NOT TARGET ninfer::json)
  add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::json INTERFACE "${_engine}/third_party")
endif()
if(NOT TARGET ninfer_xgrammar)
  add_subdirectory("${_engine}/third_party/xgrammar" "${CMAKE_CURRENT_BINARY_DIR}/engine-xgrammar"
    EXCLUDE_FROM_ALL)
endif()
if(NOT TARGET ninfer_jinja)
  add_subdirectory("${_engine}/third_party/llama-jinja" "${CMAKE_CURRENT_BINARY_DIR}/engine-jinja"
    EXCLUDE_FROM_ALL)
endif()

add_library(ninfer_ops_amd INTERFACE)
target_link_libraries(ninfer_ops_amd INTERFACE ${_engine_required_ops})

ninfer_engine_source_paths(text _engine_text_sources)
add_library(ninfer_text_amd STATIC ${_engine_text_sources})
set_target_properties(ninfer_text_amd PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
ninfer_native_hip(ninfer_text_amd)
target_compile_definitions(ninfer_text_amd PUBLIC UTF8PROC_STATIC)
target_link_libraries(ninfer_text_amd PRIVATE ninfer_xgrammar ninfer::json)

ninfer_engine_source_paths(jinja_bridge _engine_jinja_bridge_sources)
target_sources(ninfer_jinja PRIVATE ${_engine_jinja_bridge_sources})
ninfer_native_hip(ninfer_jinja)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text_amd ninfer::json)

set(_ffmpeg_default "")
set(_ffmpeg_receipt "${CMAKE_CURRENT_LIST_DIR}/../../../../research/ninfer-ffmpeg-dependency.json")
if(EXISTS "${_ffmpeg_receipt}")
  file(READ "${_ffmpeg_receipt}" _ffmpeg_receipt_json)
  string(JSON _ffmpeg_default GET "${_ffmpeg_receipt_json}" extracted_root)
endif()
set(NINFER_AMD_FFMPEG_ROOT "${_ffmpeg_default}" CACHE PATH "Pinned FFmpeg development package root")
if(NOT EXISTS "${NINFER_AMD_FFMPEG_ROOT}/include/libavformat/avformat.h")
  message(FATAL_ERROR "NINFER_AMD_FFMPEG_ROOT must contain the real FFmpeg headers and import libraries")
endif()
add_library(ninfer_ffmpeg_amd INTERFACE)
target_include_directories(ninfer_ffmpeg_amd INTERFACE "${NINFER_AMD_FFMPEG_ROOT}/include")
foreach(_library avformat avcodec avutil swscale)
  set(_import_library "${NINFER_AMD_FFMPEG_ROOT}/lib/${_library}.lib")
  if(NOT EXISTS "${_import_library}")
    message(FATAL_ERROR "Missing actual FFmpeg import library ${_import_library}")
  endif()
  target_link_libraries(ninfer_ffmpeg_amd INTERFACE "${_import_library}")
endforeach()
file(GLOB _engine_ffmpeg_dlls "${NINFER_AMD_FFMPEG_ROOT}/bin/*.dll")
function(ninfer_engine_stage_media_dlls target)
  foreach(_dll IN LISTS _engine_ffmpeg_dlls)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${_dll}" "$<TARGET_FILE_DIR:${target}>" VERBATIM)
  endforeach()
endfunction()

# This full FFmpeg package uses the original FFmpeg PNG path. The upstream
# optional native PNG implementation remains selectable with its real zlib
# dependency; it was the default workaround for a different vcpkg package.
option(NINFER_MEDIA_NATIVE_PNG "Use the original native PNG decoder instead of FFmpeg" OFF)
ninfer_engine_source_paths(media _engine_media_sources)
if(NINFER_MEDIA_NATIVE_PNG)
  ninfer_engine_source_paths(media_native_png _engine_png_sources)
  list(APPEND _engine_media_sources ${_engine_png_sources})
endif()
add_library(ninfer_media_decode_amd STATIC ${_engine_media_sources})
ninfer_native_hip(ninfer_media_decode_amd)
target_link_libraries(ninfer_media_decode_amd PRIVATE ninfer_ffmpeg_amd)
if(NINFER_MEDIA_NATIVE_PNG)
  find_package(ZLIB REQUIRED)
  target_compile_definitions(ninfer_media_decode_amd PUBLIC NINFER_MEDIA_NATIVE_PNG)
  target_link_libraries(ninfer_media_decode_amd PRIVATE ZLIB::ZLIB)
endif()

set(_flash_moe_relative
  src/models/qwen3_8_flash_next/impl/moe_kernels.cu
  src/models/qwen3_8_flash_next/impl/moe_shared_kernels.cu
  src/models/qwen3_8_flash_next/impl/moe_route.cu
  src/models/qwen3_8_flash_next/impl/stage_ledger.cpp)
set(_flash_moe_sources ${_flash_moe_relative})
list(TRANSFORM _flash_moe_sources PREPEND "${_generated}/")
add_library(ninfer_flash_next_moe_kernels_amd STATIC ${_flash_moe_sources})
ninfer_native_hip(ninfer_flash_next_moe_kernels_amd)
target_include_directories(ninfer_flash_next_moe_kernels_amd PUBLIC
  "${_generated}/src/models/qwen3_8_flash_next/export"
  "${_generated}/src/models/qwen3_vision/export")
target_link_libraries(ninfer_flash_next_moe_kernels_amd PUBLIC ninfer_basic_ops_amd)

foreach(_component model_loading model_runtime calibration engine)
  ninfer_engine_source_paths(${_component} _engine_component_sources)
  if(_component STREQUAL "model_runtime")
    # The same four original manifest TUs have one compile owner and form a
    # bounded qualification closure before the rest of the model is linked.
    list(REMOVE_ITEM _engine_component_sources ${_flash_moe_sources})
  endif()
  add_library(ninfer_${_component}_amd STATIC ${_engine_component_sources})
  ninfer_native_hip(ninfer_${_component}_amd)
  target_include_directories(ninfer_${_component}_amd PUBLIC
    "${_generated}/src/models/qwen3_8_flash_next/export"
    "${_generated}/src/models/qwen3_vision/export")
endforeach()
target_link_libraries(ninfer_model_loading_amd PUBLIC ninfer_artifact_amd ninfer_text_amd ninfer_ops_amd)
target_link_libraries(ninfer_model_runtime_amd PUBLIC ninfer_model_loading_amd ninfer_ops_amd
  ninfer_runtime_support_amd ninfer_flash_next_moe_kernels_amd PRIVATE ninfer_media_decode_amd ninfer_jinja)
target_link_libraries(ninfer_calibration_amd PUBLIC ninfer_ops_amd ninfer_core_amd)
target_include_directories(ninfer_engine_amd PRIVATE "${_engine}/third_party")
target_link_libraries(ninfer_engine_amd PRIVATE ninfer_calibration_amd ninfer_model_runtime_amd
  ninfer_artifact_amd ninfer_core_amd ninfer_runtime_support_amd ninfer_ops_amd
  ninfer_text_amd ninfer_xgrammar ninfer_media_decode_amd)
add_library(ninfer::engine ALIAS ninfer_engine_amd)

add_executable(ninfer_amd_memcpy_batch_test "${CMAKE_CURRENT_LIST_DIR}/engine/test_memcpy_batch.cpp")
ninfer_native_hip(ninfer_amd_memcpy_batch_test)
target_link_libraries(ninfer_amd_memcpy_batch_test PRIVATE ninfer_core_amd)
ninfer_add_gpu_test(ninfer_amd_memcpy_batch_test ninfer_amd_memcpy_batch_test 30)
add_executable(ninfer_amd_flash_next_moe_gate_up_test "${CMAKE_CURRENT_LIST_DIR}/engine/test_moe_gate_up.cpp")
ninfer_native_hip(ninfer_amd_flash_next_moe_gate_up_test)
target_include_directories(ninfer_amd_flash_next_moe_gate_up_test PRIVATE "${_generated}/tests")
target_compile_options(ninfer_amd_flash_next_moe_gate_up_test PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(ninfer_amd_flash_next_moe_gate_up_test PRIVATE ninfer_flash_next_moe_kernels_amd)
ninfer_add_gpu_test(ninfer_amd_flash_next_moe_gate_up_test ninfer_amd_flash_next_moe_gate_up_test 120)

# This CPU-only qualification includes the actual generated expert header.
# Keep its scalar dispatcher on the baseline ISA and isolate SIMD features in
# the production functions' local targets; no HIP runtime/link is required.
add_executable(ninfer_amd_cpu_expert_a16_test
  "${CMAKE_CURRENT_LIST_DIR}/engine/tests/cpu_expert_qualification.cpp")
target_include_directories(ninfer_amd_cpu_expert_a16_test PRIVATE "${_generated}/src")
target_compile_options(ninfer_amd_cpu_expert_a16_test PRIVATE -fno-fast-math -ffp-contract=off)
ninfer_add_gpu_command_test(ninfer_amd_cpu_expert_a16_test 180
  "$<TARGET_FILE:ninfer_amd_cpu_expert_a16_test>")
ninfer_add_gpu_command_test(ninfer_amd_cpu_expert_a16_forward_test 180
  "$<TARGET_FILE:ninfer_amd_cpu_expert_a16_test>" --forward)
set_tests_properties(ninfer_amd_cpu_expert_a16_test ninfer_amd_cpu_expert_a16_forward_test
  PROPERTIES LABELS "cpu;cpu_expert")

# Preserve every original test's source, arguments and skip contract. Tests that
# can touch device state use the shared lock; pure text/media tests stay on CPU.
string(JSON _engine_test_count LENGTH "${_engine_manifest}" tests)
math(EXPR _engine_last_test "${_engine_test_count} - 1")
foreach(_index RANGE 0 ${_engine_last_test})
  string(JSON _original MEMBER "${_engine_manifest}" tests ${_index})
  string(REGEX REPLACE "^ninfer_" "ninfer_amd_" _target "${_original}")
  set(_test_sources)
  string(JSON _count LENGTH "${_engine_manifest}" tests ${_original} sources)
  math(EXPR _last "${_count} - 1")
  foreach(_item RANGE 0 ${_last})
    string(JSON _source GET "${_engine_manifest}" tests ${_original} sources ${_item})
    list(APPEND _test_sources "${_generated}/${_source}")
  endforeach()
  add_executable(${_target} ${_test_sources})
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  string(JSON _count LENGTH "${_engine_manifest}" tests ${_original} libraries)
  if(_count GREATER 0)
    math(EXPR _last "${_count} - 1")
    foreach(_item RANGE 0 ${_last})
      string(JSON _library GET "${_engine_manifest}" tests ${_original} libraries ${_item})
      if(NOT _library STREQUAL "ninfer_jinja" AND NOT _library STREQUAL "ninfer::json")
        string(APPEND _library "_amd")
      endif()
      if(NOT TARGET ${_library})
        message(FATAL_ERROR "Unclosed original Engine test dependency: ${_original}: ${_library}")
      endif()
      target_link_libraries(${_target} PRIVATE ${_library})
    endforeach()
  endif()
  string(JSON _needs_source GET "${_engine_manifest}" tests ${_original} needs_source_dir)
  if(_needs_source)
    target_compile_definitions(${_target} PRIVATE NINFER_SOURCE_DIR="${_engine}"
      NINFER_PYTHON_EXECUTABLE="${NINFER_PYTHON}")
  endif()
  set(_test_arguments)
  string(JSON _count LENGTH "${_engine_manifest}" tests ${_original} arguments)
  if(_count GREATER 0)
    math(EXPR _last "${_count} - 1")
    foreach(_item RANGE 0 ${_last})
      string(JSON _argument GET "${_engine_manifest}" tests ${_original} arguments ${_item})
      list(APPEND _test_arguments "${_argument}")
    endforeach()
  endif()
  if(_original MATCHES "^ninfer_(jinja|structured_output|media_decode)_test$")
    add_test(NAME ${_target} COMMAND ${_target} ${_test_arguments})
    set_tests_properties(${_target} PROPERTIES TIMEOUT 60)
  else()
    ninfer_add_gpu_test(${_target} ${_target} 1800 ${_test_arguments})
  endif()
  string(JSON _skip_type TYPE "${_engine_manifest}" tests ${_original} skip_return_code)
  if(NOT _skip_type STREQUAL "NULL")
    string(JSON _skip GET "${_engine_manifest}" tests ${_original} skip_return_code)
    set_tests_properties(${_target} PROPERTIES SKIP_RETURN_CODE ${_skip})
  endif()
  ninfer_engine_stage_media_dlls(${_target})
endforeach()
add_test(NAME ninfer_amd_chat_templates_test COMMAND "${NINFER_PYTHON}" -B
  "${_engine}/tests/text/test_chat_templates.py" "$<TARGET_FILE:ninfer_amd_jinja_test>")
set_tests_properties(ninfer_amd_chat_templates_test PROPERTIES TIMEOUT 120)
ninfer_add_gpu_test(ninfer_amd_ngram_graph_planning_real ninfer_amd_ngram_graph_planning_test 1800 --real)
set_tests_properties(ninfer_amd_ngram_graph_planning_real PROPERTIES SKIP_RETURN_CODE 77)
