# Build the real v3 artifact loader/materializer. Its source ownership and stored
# formats remain those of the upstream component; no model-inference substitute.
file(READ "${_engine}/src/artifact/CMakeLists.txt" _artifact_manifest)
string(REGEX MATCHALL "\n[ \t]+[A-Za-z_]+\\.(cu|cpp)" _artifact_names "${_artifact_manifest}")
set(_artifact_paths)
foreach(_source IN LISTS _artifact_names)
  string(STRIP "${_source}" _source)
  list(APPEND _artifact_paths "${_generated}/src/artifact/${_source}")
endforeach()
add_library(ninfer_artifact_amd STATIC ${_artifact_paths})
ninfer_native_hip(ninfer_artifact_amd)
target_include_directories(ninfer_artifact_amd PUBLIC "${_engine}/third_party")
target_link_libraries(ninfer_artifact_amd PUBLIC ninfer_core_amd)

# All original reader/binder assertions are CPU-only; do not occupy the GPU lock.
add_executable(ninfer_amd_artifact_reader_test "${_generated}/tests/artifact/test_reader.cpp")
ninfer_native_hip(ninfer_amd_artifact_reader_test)
target_link_libraries(ninfer_amd_artifact_reader_test PRIVATE ninfer_artifact_amd)
target_include_directories(ninfer_amd_artifact_reader_test PRIVATE "${_generated}/tests")
add_test(NAME ninfer_amd_artifact_reader_test COMMAND ninfer_amd_artifact_reader_test)
set_tests_properties(ninfer_amd_artifact_reader_test PROPERTIES TIMEOUT 60)

foreach(_test permute_row_split materialization)
  set(_target "ninfer_amd_artifact_${_test}_test")
  add_executable(${_target} "${_generated}/tests/artifact/test_${_test}.cpp")
  ninfer_native_hip(${_target})
  target_link_libraries(${_target} PRIVATE ninfer_artifact_amd)
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  # Windows has no GNU link wrapping. Match upstream's Windows materialization
  # test: no NINFER_TEST_LINK_WRAP, keeping its real upload/restore checks.
  ninfer_add_gpu_test(${_target} ${_target} 60)
endforeach()

# Use the unchanged production Python writer against the HIP materializer.
# test_locked owns the entire Python child tree, including its two GPU children.
ninfer_add_gpu_command_test(ninfer_amd_artifact_writer_interop_test 60
  "${NINFER_PYTHON}" -B "${_engine}/tests/artifact/writer_interop.py"
  "$<TARGET_FILE:ninfer_amd_artifact_materialization_test>")
