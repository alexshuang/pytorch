set(TORCH_TRACE_TEST_DIR "${TORCH_ROOT}/test/cpp/trace")
set(TORCH_TRACE_TEST_SOURCES
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TORCH_TRACE_TEST_DIR}/tensor.cpp
)

if(USE_CUDA)
  list(APPEND TORCH_TRACE_TEST_SOURCES ${TORCH_TRACE_TEST_DIR}/parallel.cpp)
endif()

add_executable(test_trace ${TORCH_TRACE_TEST_SOURCES})
target_include_directories(test_trace PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(test_trace PRIVATE torch gtest)

if(USE_CUDA)
  target_link_libraries(test_trace PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_trace PRIVATE "USE_CUDA")
endif()

if(NOT MSVC)
  if(APPLE)
    target_compile_options(test_trace PRIVATE
      # Clang has an unfixed bug leading to spurious missing braces
      # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
      -Wno-missing-braces)
  else()
    target_compile_options(test_trace PRIVATE
      # Considered to be flaky.  See the discussion at
      # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized
      # gcc gives nonsensical warnings about variadic.h
      -Wno-unused-but-set-parameter)
  endif()
endif()

if(INSTALL_TEST)
  install(TARGETS test_trace DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_trace> DESTINATION bin OPTIONAL)
  endif()
endif()
