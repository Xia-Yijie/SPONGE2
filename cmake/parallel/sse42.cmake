include("${PROJECT_ROOT_DIR}/cmake/parallel/none.cmake")

if(MSVC)
  target_compile_options(common_libraries INTERFACE /arch:SSE2)
else()
  target_compile_options(common_libraries INTERFACE -msse4.2)
endif()
target_compile_definitions(common_libraries INTERFACE USE_SSE42)
