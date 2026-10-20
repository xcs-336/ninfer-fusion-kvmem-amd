# The original public products and end-to-end inference benchmark. This file
# adds compile owners; every inference command uses the existing public Engine.
if(NOT TARGET ninfer_engine_amd)
  message(FATAL_ERROR "Native products require the complete original public Engine")
endif()
set(_products_port "${CMAKE_CURRENT_LIST_DIR}/engine")
file(GLOB_RECURSE _products_inputs CONFIGURE_DEPENDS
  "${_products_port}/products/*.py" "${_products_port}/products/*.h"
  "${_products_port}/products/*.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_products_port}/products.py" ${_products_inputs})
execute_process(COMMAND "${NINFER_PYTHON}" "${_products_port}/products.py"
  --engine "${_engine}" --source-view "${_prepared}" --output "${_prepared}"
  RESULT_VARIABLE _products_status COMMAND_ECHO STDOUT)
if(NOT _products_status EQUAL 0)
  message(FATAL_ERROR "Original native product source preparation failed")
endif()
file(READ "${_prepared}/products-manifest.json" _products_manifest)
string(JSON _products_count LENGTH "${_products_manifest}" inputs)
math(EXPR _products_last "${_products_count} - 1")
foreach(_index RANGE 0 ${_products_last})
  string(JSON _input GET "${_products_manifest}" inputs ${_index})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_engine}/${_input}")
endforeach()

function(ninfer_product_sources category name output)
  string(JSON _count LENGTH "${_products_manifest}" ${category} ${name})
  math(EXPR _last "${_count} - 1")
  set(_paths)
  foreach(_index RANGE 0 ${_last})
    string(JSON _source GET "${_products_manifest}" ${category} ${name} ${_index})
    list(APPEND _paths "${_generated}/${_source}")
  endforeach()
  set(${output} "${_paths}" PARENT_SCOPE)
endfunction()

if(NOT TARGET spdlog::spdlog)
  add_subdirectory("${_engine}/third_party/spdlog" "${CMAKE_CURRENT_BINARY_DIR}/products-spdlog"
    EXCLUDE_FROM_ALL)
endif()
if(NOT TARGET ninfer::httplib)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE "${_engine}/third_party/cpp-httplib")
endif()

# A single installed, matching header/import-library/DLL package is selected.
# Do not mix its 8.18 headers with the active conda environment's older files.
set(NINFER_AMD_CURL_ROOT "D:/programfile/work/miniconda/pkgs/libcurl-8.18.0-h43ecb02_0/Library"
  CACHE PATH "Real libcurl >=7.85 development package root")
foreach(_file include/curl/curl.h include/curl/curlver.h lib/libcurl.lib bin/libcurl.dll)
  if(NOT EXISTS "${NINFER_AMD_CURL_ROOT}/${_file}")
    message(FATAL_ERROR "Missing actual libcurl development file ${NINFER_AMD_CURL_ROOT}/${_file}")
  endif()
endforeach()
file(STRINGS "${NINFER_AMD_CURL_ROOT}/include/curl/curlver.h" _curl_version_line
  REGEX "^#define LIBCURL_VERSION \"[0-9.]+\"")
string(REGEX REPLACE ".*\"([0-9.]+)\".*" "\\1" _curl_version "${_curl_version_line}")
if(_curl_version VERSION_LESS "7.85")
  message(FATAL_ERROR "Original media acquisition needs actual libcurl >=7.85")
endif()
add_library(ninfer_curl_amd SHARED IMPORTED GLOBAL)
set_target_properties(ninfer_curl_amd PROPERTIES
  IMPORTED_IMPLIB "${NINFER_AMD_CURL_ROOT}/lib/libcurl.lib"
  IMPORTED_LOCATION "${NINFER_AMD_CURL_ROOT}/bin/libcurl.dll"
  INTERFACE_INCLUDE_DIRECTORIES "${NINFER_AMD_CURL_ROOT}/include")
set(NINFER_AMD_CURL_DEPENDENCY_BIN "D:/programfile/work/miniconda/Library/bin" CACHE PATH
  "Actual installed libcurl transitive runtime DLL directory")
execute_process(COMMAND "${NINFER_PYTHON}" "${_products_port}/products/dependency_dlls.py"
  --readobj "${ROCM_ROOT}/lib/llvm/bin/llvm-readobj.exe"
  --entry "${NINFER_AMD_CURL_ROOT}/bin/libcurl.dll"
  --root "${NINFER_AMD_CURL_ROOT}/bin" --root "${NINFER_AMD_CURL_DEPENDENCY_BIN}"
  --output "${_prepared}/products-dll-manifest.json"
  RESULT_VARIABLE _products_dll_status COMMAND_ECHO STDOUT)
if(NOT _products_dll_status EQUAL 0)
  message(FATAL_ERROR "Actual libcurl runtime DLL dependency closure is incomplete")
endif()
file(READ "${_prepared}/products-dll-manifest.json" _products_dll_manifest)
string(JSON _products_dll_count LENGTH "${_products_dll_manifest}" dlls)
math(EXPR _products_last_dll "${_products_dll_count} - 1")
set(_products_dlls ${_engine_ffmpeg_dlls})
foreach(_index RANGE 0 ${_products_last_dll})
  string(JSON _dll GET "${_products_dll_manifest}" dlls ${_index} source)
  list(APPEND _products_dlls "${_dll}")
endforeach()
function(ninfer_product_stage_runtime target)
  foreach(_dll IN LISTS _products_dlls)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${_dll}" "$<TARGET_FILE_DIR:${target}>" VERBATIM)
  endforeach()
endfunction()

foreach(_component media_acquire product_prompt_input product_logging serve)
  ninfer_product_sources(components ${_component} _sources)
  add_library(ninfer_${_component}_amd STATIC ${_sources})
  ninfer_native_hip(ninfer_${_component}_amd)
endforeach()
target_link_libraries(ninfer_media_acquire_amd PRIVATE ninfer_curl_amd ws2_32)
target_link_libraries(ninfer_product_prompt_input_amd PRIVATE ninfer_media_acquire_amd ninfer::json)
target_link_libraries(ninfer_product_logging_amd PUBLIC spdlog::spdlog)
target_link_libraries(ninfer_serve_amd PUBLIC ninfer_engine_amd Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire_amd ninfer_product_logging_amd ninfer_product_prompt_input_amd)

set(NINFER_WEBUI_DIR "" CACHE PATH "Directory of original built WebUI files; empty embeds none")
if(NINFER_WEBUI_DIR)
  if(NOT EXISTS "${NINFER_WEBUI_DIR}/index.html" AND NOT EXISTS "${NINFER_WEBUI_DIR}/index.html.gz")
    message(FATAL_ERROR "NINFER_WEBUI_DIR has no index.html: ${NINFER_WEBUI_DIR}")
  endif()
  string(JSON _embed_source GET "${_products_manifest}" webui embed)
  add_executable(ninfer_amd_webui_embed "${_generated}/${_embed_source}")
  file(GLOB_RECURSE _webui_files CONFIGURE_DEPENDS "${NINFER_WEBUI_DIR}/*")
  set(_webui_source "${CMAKE_CURRENT_BINARY_DIR}/products-webui-assets.cpp")
  add_custom_command(OUTPUT "${_webui_source}"
    COMMAND ninfer_amd_webui_embed "${_webui_source}" "${NINFER_WEBUI_DIR}"
    DEPENDS ninfer_amd_webui_embed ${_webui_files} VERBATIM)
  target_sources(ninfer_serve_amd PRIVATE "${_webui_source}")
else()
  string(JSON _none_source GET "${_products_manifest}" webui none)
  target_sources(ninfer_serve_amd PRIVATE "${_generated}/${_none_source}")
endif()

set(_products_build_id_header "${_generated}/ninfer_build_id.h")
add_custom_target(ninfer_amd_build_id_stamp ALL
  COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${_engine}" "-DOUT=${_products_build_id_header}"
    -P "${_generated}/cmake/GenerateBuildId.cmake"
  BYPRODUCTS "${_products_build_id_header}" VERBATIM)
foreach(_app ninfer ninfer-serve ninfer-calibrate ninfer-perplexity ninfer_bench)
  ninfer_product_sources(executables ${_app} _sources)
  add_executable(${_app} ${_sources})
  ninfer_native_hip(${_app})
  if(NOT _app STREQUAL "ninfer-calibrate" AND NOT _app STREQUAL "ninfer_bench")
    add_dependencies(${_app} ninfer_amd_build_id_stamp)
    target_sources(${_app} PRIVATE "${_products_build_id_header}")
    target_include_directories(${_app} PRIVATE "${_generated}")
  endif()
  ninfer_product_stage_runtime(${_app})
endforeach()
target_link_libraries(ninfer PRIVATE ninfer_engine_amd ninfer_product_logging_amd ninfer_product_prompt_input_amd)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve_amd ninfer_product_logging_amd)
target_link_libraries(ninfer-calibrate PRIVATE ninfer_calibration_amd ninfer_runtime_support_amd)
target_link_libraries(ninfer-perplexity PRIVATE ninfer_engine_amd ninfer_product_logging_amd ninfer::json)
target_include_directories(ninfer_bench PRIVATE "${_generated}/bench")
target_compile_definitions(ninfer_bench PRIVATE NINFER_SOURCE_DIR="${_engine}")
target_link_libraries(ninfer_bench PRIVATE ninfer_engine_amd)

string(JSON _products_test_count LENGTH "${_products_manifest}" tests)
math(EXPR _products_last_test "${_products_test_count} - 1")
foreach(_index RANGE 0 ${_products_last_test})
  string(JSON _original MEMBER "${_products_manifest}" tests ${_index})
  string(REGEX REPLACE "^ninfer_" "ninfer_amd_" _target "${_original}")
  if(TARGET ${_target})
    continue() # media_decode is already owned by the original Engine closure
  endif()
  ninfer_product_sources(tests "${_original};sources" _sources)
  add_executable(${_target} ${_sources})
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  string(JSON _count LENGTH "${_products_manifest}" tests ${_original} include_directories)
  if(_count GREATER 0)
    math(EXPR _last "${_count} - 1")
    foreach(_item RANGE 0 ${_last})
      string(JSON _directory GET "${_products_manifest}" tests ${_original} include_directories ${_item})
      target_include_directories(${_target} PRIVATE "${_generated}/${_directory}")
    endforeach()
  endif()
  string(JSON _count LENGTH "${_products_manifest}" tests ${_original} libraries)
  if(_count GREATER 0)
    math(EXPR _last "${_count} - 1")
    foreach(_item RANGE 0 ${_last})
      string(JSON _library GET "${_products_manifest}" tests ${_original} libraries ${_item})
      if(NOT _library STREQUAL "ninfer::json")
        string(APPEND _library "_amd")
      endif()
      if(NOT TARGET ${_library})
        message(FATAL_ERROR "Unclosed original product test dependency: ${_original}: ${_library}")
      endif()
      target_link_libraries(${_target} PRIVATE ${_library})
    endforeach()
  endif()
  string(JSON _needs_source GET "${_products_manifest}" tests ${_original} needs_source_dir)
  if(_needs_source)
    target_compile_definitions(${_target} PRIVATE NINFER_SOURCE_DIR="${_engine}"
      NINFER_PYTHON_EXECUTABLE="${NINFER_PYTHON}")
  endif()
  set(_arguments)
  string(JSON _count LENGTH "${_products_manifest}" tests ${_original} arguments)
  if(_count GREATER 0)
    math(EXPR _last "${_count} - 1")
    foreach(_item RANGE 0 ${_last})
      string(JSON _argument GET "${_products_manifest}" tests ${_original} arguments ${_item})
      list(APPEND _arguments "${_argument}")
    endforeach()
  endif()
  add_test(NAME ${_target} COMMAND ${_target} ${_arguments})
  set_tests_properties(${_target} PROPERTIES TIMEOUT 120
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")
  ninfer_product_stage_runtime(${_target})
endforeach()

add_executable(ninfer_amd_product_environment_test
  "${_products_port}/products/test_native_environment.cpp"
  "${_generated}/bench/inference/ninfer_bench_support.cpp")
ninfer_native_hip(ninfer_amd_product_environment_test)
target_include_directories(ninfer_amd_product_environment_test PRIVATE "${_generated}/bench/inference")
target_link_libraries(ninfer_amd_product_environment_test PRIVATE ninfer_serve_amd ninfer::json)
add_test(NAME ninfer_amd_product_environment_test COMMAND ninfer_amd_product_environment_test)
set_tests_properties(ninfer_amd_product_environment_test PROPERTIES TIMEOUT 30
  ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")
ninfer_product_stage_runtime(ninfer_amd_product_environment_test)

# Keep ROCm's actual DLL and kernel-data layout together in the installed SDK.
# This shell environment executes the unchanged original product binaries.
file(TO_NATIVE_PATH "${ROCM_ROOT}/bin" _products_sdk_bin)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ninfer-amd-env.cmd" CONTENT
  "@echo off\nsetlocal\nset \"PATH=${_products_sdk_bin};%PATH%\"\n%*\nexit /b %errorlevel%\n")
foreach(_app ninfer ninfer-serve ninfer-calibrate ninfer-perplexity ninfer_bench)
  string(REPLACE "-" "_" _app_test "${_app}")
  add_test(NAME ninfer_amd_${_app_test}_help COMMAND ${_app} --help)
  set_tests_properties(ninfer_amd_${_app_test}_help PROPERTIES TIMEOUT 30
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")
endforeach()

# These actual application error/help paths return before any Engine, artifact,
# profile lookup or GPU initialization. No mirrored perplexity parser is used.
add_test(NAME ninfer_amd_device_profile_cli_contract
  COMMAND "${NINFER_PYTHON}" -B "${_products_port}/products/test_profile_cli_contract.py"
    --cli "$<TARGET_FILE:ninfer>" --bench "$<TARGET_FILE:ninfer_bench>"
    --perplexity "$<TARGET_FILE:ninfer-perplexity>")
set_tests_properties(ninfer_amd_device_profile_cli_contract PROPERTIES TIMEOUT 100
  ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")
