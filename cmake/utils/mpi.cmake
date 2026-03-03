message(STATUS "Load MPI options from ${MPI_FILE}")

set(MPI
    off
    CACHE BOOL "use MPI")
message(STATUS "-- MPI switch: ${MPI}")

if(MPI)
  add_definitions(-DUSE_MPI)
  find_package(MPI)
  target_link_libraries(common_libraries INTERFACE MPI::MPI_CXX)
  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "MPI option enabled, but MPI library not found")
  endif()
  if(PARALLEL_BACKEND STREQUAL "cuda")
    find_package(NCCL)
    if(NOT NCCL_FOUND)
      message(
        FATAL_ERROR
          "MPI option enabled and CUDA used as parallel backend, but NCCL library not found"
      )
    endif()
    # propagate NCCL include dir and libraries to common_libraries so targets
    # link properly
    if(NCCL_FOUND)
      if(DEFINED NCCL_INCLUDE_DIRS)
        target_include_directories(common_libraries
                                   INTERFACE ${NCCL_INCLUDE_DIRS})
      endif()
      if(DEFINED NCCL_LIBRARIES)
        target_link_libraries(common_libraries INTERFACE ${NCCL_LIBRARIES})
      endif()
    endif()
  endif()
endif()

message(
  STATUS "------------------------------------------------------------------")
