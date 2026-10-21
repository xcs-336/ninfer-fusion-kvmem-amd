# NEW, unwired leaf. Root may include after the NEW controller full-Core leaf.
option(NINFER_AMD_MODELS_SEGMENTED_STORAGE "Build isolated two-file Models storage candidate" OFF)
if(NOT NINFER_AMD_MODELS_SEGMENTED_STORAGE)
  return()
endif()
if(NOT TARGET ninfer_controller_handoff_core_amd OR NOT COMMAND ninfer_controller_handoff_native OR
   NOT DEFINED NINFER_AMD_CONTROLLER_HANDOFF_VIEW OR NOT DEFINED _generated OR NOT DEFINED NINFER_PYTHON)
  message(FATAL_ERROR "Models storage requires the NEW full controller Core ABI, never the old partial Core archive")
endif()
set(_models_storage_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_models_storage_inputs "${CMAKE_CURRENT_LIST_FILE}"
  "${_models_storage_dir}/prepare_models_storage.py" "${_models_storage_dir}/decoder_storage_impl.cpp.inc"
  "${_models_storage_dir}/test_models_storage.cu" "${_models_storage_dir}/verify_models_storage_map.py"
  "${_generated}/src/models/qwen3_5/state/decoder_state.h"
  "${_generated}/src/models/qwen3_5/state/decoder_state.cpp"
  "${_generated}/src/ops/kv_cache/d256_profile.h"
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/controller-handoff-source-receipt.json"
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src/core/paged_kv_cache.h"
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/amd-memory/kv_segmented_pool_owner.h")
set(_models_storage_recipe "two-real-DecoderState-files-single-external-payload-v1\n")
foreach(_file IN LISTS _models_storage_inputs)
  file(SHA256 "${_file}" _digest)
  string(APPEND _models_storage_recipe "${_file}:${_digest}\n")
endforeach()
string(SHA256 _models_storage_sha "${_models_storage_recipe}")
set(NINFER_AMD_MODELS_STORAGE_VIEW "${CMAKE_CURRENT_BINARY_DIR}/models-storage-candidate/${_models_storage_sha}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${_models_storage_dir}/prepare_models_storage.py"
  --native-root "${_generated}" --core-root "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}"
  --output "${NINFER_AMD_MODELS_STORAGE_VIEW}" --verify-existing
  RESULT_VARIABLE _models_storage_prepared ERROR_VARIABLE _models_storage_error)
if(NOT _models_storage_prepared EQUAL 0)
  message(FATAL_ERROR "Models storage isolation refused: ${_models_storage_error}")
endif()
add_library(ninfer_models_storage_candidate_amd STATIC EXCLUDE_FROM_ALL
  "${NINFER_AMD_MODELS_STORAGE_VIEW}/src/models/qwen3_5/state/decoder_state.cpp")
ninfer_controller_handoff_native(ninfer_models_storage_candidate_amd)
target_include_directories(ninfer_models_storage_candidate_amd BEFORE PUBLIC
  "${NINFER_AMD_MODELS_STORAGE_VIEW}/src" "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src"
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/amd-memory" "${_generated}/src")
target_link_libraries(ninfer_models_storage_candidate_amd PUBLIC ninfer_controller_handoff_core_amd)
add_executable(ninfer_amd_models_segmented_storage_test EXCLUDE_FROM_ALL "${_models_storage_dir}/test_models_storage.cu")
ninfer_controller_handoff_native(ninfer_amd_models_segmented_storage_test)
target_include_directories(ninfer_amd_models_segmented_storage_test BEFORE PRIVATE
  "${NINFER_AMD_MODELS_STORAGE_VIEW}/src" "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src"
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/amd-memory" "${_generated}/src")
target_link_libraries(ninfer_amd_models_segmented_storage_test PRIVATE ninfer_models_storage_candidate_amd)
set(_models_storage_map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_models_segmented_storage_test.map")
target_link_options(ninfer_amd_models_segmented_storage_test PRIVATE "-Wl,/MAP:${_models_storage_map}")
add_custom_command(TARGET ninfer_amd_models_segmented_storage_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_models_storage_dir}/verify_models_storage_map.py" --map "${_models_storage_map}"
  VERBATIM COMMENT "Require new Models/full-Core extraction and reject all legacy layout archives")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_models_storage_inputs})
# No Core.cmake edit, Engine wiring, old Models/Core library, whole-archive, or automatic GPU CTest.
