file(READ "${_engine}/src/ops/linear/gguf/sources.cmake" _gguf_manifest)
string(REGEX MATCHALL "[A-Za-z_0-9]+\\.cu\"" _gguf_sources "${_gguf_manifest}")
set(_gguf_paths)
foreach(_source IN LISTS _gguf_sources)
  string(REPLACE "\"" "" _source "${_source}")
  list(APPEND _gguf_paths "${_generated}/src/ops/linear/gguf/${_source}")
endforeach()
add_library(ninfer_ggml_quants_amd STATIC ${_gguf_paths})
ninfer_native_hip(ninfer_ggml_quants_amd)
target_compile_definitions(ninfer_ggml_quants_amd PRIVATE GGML_USE_HIP=1 GGML_HIP_NO_VMM=1)
target_include_directories(ninfer_ggml_quants_amd PRIVATE
  "${_generated}/src/ops/linear/gguf" "${_engine}/third_party/ggml-quants/include"
  "${_engine}/third_party/ggml-quants/src")
target_link_libraries(ninfer_ggml_quants_amd PUBLIC "${ROCM_ROOT}/lib/hipblas.lib")
target_sources(ninfer_basic_ops_amd PRIVATE
  "${_generated}/src/ops/linear/gguf/gguf_linear.cpp"
  "${_generated}/src/ops/linear/fp8/fp8_format.cpp")
target_link_libraries(ninfer_basic_ops_amd PUBLIC ninfer_ggml_quants_amd)
