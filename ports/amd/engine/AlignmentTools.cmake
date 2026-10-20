# Optional research observers over actual production implementations. They do
# not add a second model execution path or modify the public product protocols.
if(NOT TARGET ninfer_engine_amd OR NOT TARGET ninfer_product_prompt_input_amd)
  message(FATAL_ERROR "Native alignment tools require the complete original Engine/products")
endif()

add_executable(ninfer_amd_frontend_export "${CMAKE_CURRENT_LIST_DIR}/tools/frontend_export.cpp")
ninfer_native_hip(ninfer_amd_frontend_export)
target_link_libraries(ninfer_amd_frontend_export PRIVATE
  ninfer_engine_amd ninfer_core_amd ninfer_product_prompt_input_amd ninfer::json)
# frontend_export includes the generated production frontend.cpp in its own
# TU to observe its actual renderer. Do not also list that source here.
target_compile_options(ninfer_amd_frontend_export PRIVATE -fno-fast-math -ffp-contract=off)
ninfer_product_stage_runtime(ninfer_amd_frontend_export)

add_executable(ninfer_amd_score_tokens "${CMAKE_CURRENT_LIST_DIR}/tools/score_tokens.cpp")
ninfer_native_hip(ninfer_amd_score_tokens)
target_link_libraries(ninfer_amd_score_tokens PRIVATE ninfer_engine_amd ninfer::json)
target_compile_options(ninfer_amd_score_tokens PRIVATE -fno-fast-math -ffp-contract=off)
ninfer_product_stage_runtime(ninfer_amd_score_tokens)

# These two checks only parse --help; they never construct an Engine or access
# a model. Actual frontend/scoring qualification requires explicit fixtures and
# (for scoring) the shared GPU lock and live memory guard outside CTest.
foreach(_target ninfer_amd_frontend_export ninfer_amd_score_tokens)
  add_test(NAME ${_target}_help COMMAND ${_target} --help)
  set_tests_properties(${_target}_help PROPERTIES TIMEOUT 30 LABELS "cpu;alignment_tools")
endforeach()
