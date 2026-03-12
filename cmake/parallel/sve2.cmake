include("${PROJECT_ROOT_DIR}/cmake/parallel/none.cmake")

if(MSVC)
  target_compile_options(common_libraries INTERFACE /arch:SVE)
else()
  target_compile_options(common_libraries INTERFACE -march=armv8-a+sve2)
endif()
target_compile_definitions(common_libraries INTERFACE USE_SVE2)
