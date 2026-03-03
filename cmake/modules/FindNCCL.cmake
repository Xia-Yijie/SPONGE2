# FindNCCL.cmake - robust NCCL finder for SPONGE Adapted to search common
# prefixes, support optional versioned libs, and expose NCCL_INCLUDE_DIRS and
# NCCL_LIBRARIES for CMake.

set(NCCL_INCLUDE_DIR
    $ENV{NCCL_INCLUDE_DIR}
    CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR
    $ENV{NCCL_LIB_DIR}
    CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION
    $ENV{NCCL_VERSION}
    CACHE STRING "Version of NCCL to build with")

if(DEFINED ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()

# Candidate prefixes to search
set(_NCCL_CANDIDATE_PREFIXES
    ${NCCL_LIB_DIR} ${NCCL_INCLUDE_DIR} /usr/local/nccl /usr/local /usr
    ${CUDA_TOOLKIT_ROOT_DIR})

# find include
find_path(
  NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${_NCCL_CANDIDATE_PREFIXES}
  PATH_SUFFIXES include include/nccl)

# decide library name
if(USE_STATIC_NCCL)
  set(_NCCL_LIBNAME nccl_static)
else()
  set(_NCCL_LIBNAME nccl)
endif()

# if specific version requested, prefer versioned suffix; save/restore
# CMAKE_FIND_LIBRARY_SUFFIXES
if(NCCL_VERSION)
  set(_OLD_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(USE_STATIC_NCCL)
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 ".a.${NCCL_VERSION}")
  else()
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 ".so.${NCCL_VERSION}")
  endif()
endif()

# find library
find_library(
  NCCL_LIBRARIES
  NAMES ${_NCCL_LIBNAME} nccl
  HINTS ${_NCCL_CANDIDATE_PREFIXES}
  PATH_SUFFIXES lib lib64 lib/nccl)

# restore suffixes
if(DEFINED _OLD_CMAKE_FIND_LIBRARY_SUFFIXES)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_OLD_CMAKE_FIND_LIBRARY_SUFFIXES})
  unset(_OLD_CMAKE_FIND_LIBRARY_SUFFIXES)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS
                                  NCCL_LIBRARIES)

if(NCCL_FOUND)
  message(
    STATUS
      "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  # optional: check header/library version compatibility
  set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  set(OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS})
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(NCCL_VERSION_CODE nccl.h NCCL_VERSION_DEFINED)
  if(NCCL_VERSION_DEFINED AND NCCL_LIBRARIES)
    # attempt a tiny run to verify linking (non-fatal)
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
    file(
      WRITE ${file}
      "#include <iostream>\n#include <nccl.h>\nint main(){ int v=0; ncclGetVersion(&v); std::cout<<v; return 0; }"
    )
    try_run(
      NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
      RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${NCCL_INCLUDE_DIRS}" LINK_LIBRARIES
                  ${NCCL_LIBRARIES})
    if(NOT NCCL_VERSION_MATCHED)
      message(
        WARNING
          "Found NCCL header and library but test run failed. Include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}"
      )
    endif()
  endif()
  set(CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})

  mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES NCCL_INCLUDE_DIR
                   NCCL_LIB_DIR)
else()
  message(
    STATUS
      "NCCL not found. If you need GPU multi-process communication, set NCCL_INCLUDE_DIR and NCCL_LIB_DIR or install libnccl-dev."
  )
endif()
