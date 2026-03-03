set(ARMPL_ROOT $ENV{ARMPL_ROOT} $ENV{ARMPL_DIR} /opt/arm)

set(ARMPL_COMPONENTS FFTW)
if(NOT ARMPL_FIND_COMPONENTS)
  set(ARMPL_FIND_COMPONENTS FFTW)
endif()

find_library(
  ARMPL_FFT_LIBRARIES
  NAMES "armpl_lp64"
  HINTS ${ARMPL_ROOT}
  PATH_SUFFIXES "lib" "lib64")

find_path(
  ARMPL_INCLUDE_DIRS
  NAMES "fftw3.h"
  HINTS ${ARMPL_ROOT}
  PATH_SUFFIXES "include" "include/fftw" "fftw")

# Use standard CMake package handler
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARMPL DEFAULT_MSG ARMPL_INCLUDE_DIRS
                                  ARMPL_FFT_LIBRARIES)

if(ARMPL_FOUND AND NOT TARGET ARMPL::FFTW)
  add_library(ARMPL::FFTW INTERFACE IMPORTED)
  set_target_properties(
    ARMPL::FFTW
    PROPERTIES INTERFACE_LINK_LIBRARIES "${ARMPL_FFT_LIBRARIES}"
               INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIRS}")
endif()
