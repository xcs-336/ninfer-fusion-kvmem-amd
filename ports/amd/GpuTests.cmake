set(NINFER_PYTHON "${CMAKE_CURRENT_LIST_DIR}/../../../../runtime/python311-ninfer/python.exe"
  CACHE FILEPATH "Python 3.11 for HIP generation and GPU test ownership")
if(NOT EXISTS "${NINFER_PYTHON}")
  find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Interpreter)
  set(NINFER_PYTHON "${Python3_EXECUTABLE}")
endif()
set(NINFER_GPU_LOCK_FILE "${CMAKE_CURRENT_LIST_DIR}/../../.cache/gpu-benchmark.lock"
  CACHE FILEPATH "Shared byte-0 Windows GPU lock; use one path across collaborating projects")
set(NINFER_GPU_LOCK_WAIT_SECONDS 300 CACHE STRING "Maximum time to await a shared GPU slot")
function(ninfer_add_gpu_command_test name timeout)
  add_test(NAME ${name} COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/test_locked.py"
    --lock-file "${NINFER_GPU_LOCK_FILE}" --wait-seconds "${NINFER_GPU_LOCK_WAIT_SECONDS}"
    --timeout "${timeout}" -- ${ARGN})
  # Give the owner time to await its slot and kill/reap its own job on timeout.
  math(EXPR outer_timeout "${timeout} + ${NINFER_GPU_LOCK_WAIT_SECONDS} + 15")
  set_tests_properties(${name} PROPERTIES TIMEOUT ${outer_timeout} SKIP_RETURN_CODE 77)
endfunction()
function(ninfer_add_gpu_test name target timeout)
  ninfer_add_gpu_command_test(${name} ${timeout} "$<TARGET_FILE:${target}>" ${ARGN})
endfunction()
