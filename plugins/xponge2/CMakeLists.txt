cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME _core)
project(${PROJECT_NAME} LANGUAGES C CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.SABIModule)

foreach(flag_var CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS
                 CMAKE_SHARED_LINKER_FLAGS)
  string(REGEX REPLACE "-Wl,-rpath,[^ ]+" "" ${flag_var} "${${flag_var}}")
  string(REGEX REPLACE "-Wl,-rpath-link,[^ ]+" "" ${flag_var}
                       "${${flag_var}}")
endforeach()

set(XPONGE2_VERSION "0.0.1")
set(XPONGE2_INSTALL_DIR xponge2)
set(SPONGE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../SPONGE)
set(XPONGE_CORE_DIR ${SPONGE_ROOT_DIR}/xponge)

# xponge2 is the Python ABI layer. The modeling implementation lives under
# SPONGE/xponge, and this build is intentionally rooted at SPONGE_ROOT_DIR so a
# future CPU-only SPONGE core can be compiled into the same extension without
# changing the plugin/package boundary.
set(XPONGE2_SOURCES
    src/core.cpp
    ${XPONGE_CORE_DIR}/model.cpp
    ${XPONGE_CORE_DIR}/assign/assignment.cpp
    ${XPONGE_CORE_DIR}/assign/gaff_typing.cpp
    ${XPONGE_CORE_DIR}/assign/mol2_reader.cpp
    ${XPONGE_CORE_DIR}/assign/mol2_writer_policy.cpp
    ${XPONGE_CORE_DIR}/forcefield/amber/parameters.cpp
    ${XPONGE_CORE_DIR}/forcefield/amber/parmchk2.cpp
    ${XPONGE_CORE_DIR}/forcefield/amber/sponge_writer.cpp)

add_library(${PROJECT_NAME} MODULE ${XPONGE2_SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE XPONGE2_VERSION="${XPONGE2_VERSION}"
                                   Py_LIMITED_API=0x030A0000)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${XPONGE_CORE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${XPONGE_CORE_DIR}/assign)
target_include_directories(${PROJECT_NAME} PRIVATE ${XPONGE_CORE_DIR}/forcefield/amber)
target_link_libraries(${PROJECT_NAME} PRIVATE Python3::SABIModule)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES OUTPUT_NAME ${PROJECT_NAME}
             PREFIX ""
             SUFFIX ".abi3.so")

install(
  TARGETS ${PROJECT_NAME}
  EXCLUDE_FROM_ALL
  COMPONENT python_modules
  DESTINATION ${XPONGE2_INSTALL_DIR})

install(
  DIRECTORY xponge2/
  EXCLUDE_FROM_ALL
  COMPONENT python_modules
  DESTINATION ${XPONGE2_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyi"
  PATTERN "py.typed"
  PATTERN "*.txt"
  PATTERN "*.dat"
  PATTERN "*.DAT"
  PATTERN "*.TPL")
