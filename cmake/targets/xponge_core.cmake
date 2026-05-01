set(XPONGE_CORE_SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/pyinterface.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/model.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/assign/assignment.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/assign/gaff_typing.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/assign/mol2_reader.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/assign/mol2_writer_policy.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/forcefield/amber/parameters.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/forcefield/amber/parmchk2.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/forcefield/amber/sponge_writer.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.SABIModule)

add_library(${CURRENT_TARGET} MODULE ${XPONGE_CORE_SOURCES})
target_compile_features(${CURRENT_TARGET} PRIVATE cxx_std_17)
target_compile_definitions(
  ${CURRENT_TARGET}
  PRIVATE XPONGE2_VERSION="${SPONGE_VERSION}" Py_LIMITED_API=0x030A0000)
target_include_directories(${CURRENT_TARGET} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${CURRENT_TARGET}
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE)
target_include_directories(${CURRENT_TARGET}
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE/xponge)
target_include_directories(${CURRENT_TARGET}
                           PRIVATE ${PROJECT_ROOT_DIR}/SPONGE/xponge/assign)
target_include_directories(
  ${CURRENT_TARGET}
  PRIVATE ${PROJECT_ROOT_DIR}/SPONGE/xponge/forcefield/amber)
target_link_libraries(${CURRENT_TARGET} PRIVATE Python3::SABIModule)
set_target_properties(
  ${CURRENT_TARGET}
  PROPERTIES OUTPUT_NAME "_core"
             PREFIX ""
             SUFFIX ".abi3.so")
