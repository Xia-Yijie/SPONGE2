find_package(ARMPL REQUIRED)

add_definitions(-DUSE_ARMPL)
target_link_libraries(common_libraries INTERFACE ARMPL::FFTW)
