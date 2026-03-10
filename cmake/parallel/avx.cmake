include("${PROJECT_ROOT_DIR}/cmake/parallel/none.cmake")

if(MSVC)
  target_compile_options(common_libraries INTERFACE /arch:AVX)
else()
  target_compile_options(common_libraries INTERFACE -mavx)
endif()
target_compile_definitions(common_libraries INTERFACE SPONGE_LANE_GROUP_AVX)
