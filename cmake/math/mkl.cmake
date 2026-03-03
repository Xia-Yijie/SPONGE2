find_package(MKL REQUIRED)

add_definitions(-DUSE_MKL)
include_directories(${MKL_ROOT}/include/fftw)
target_link_libraries(common_libraries INTERFACE MKL::MKL)
