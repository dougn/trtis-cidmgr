# Copyright (c) 2019, Doug Napoleone. All rights reserved.

cmake_minimum_required (VERSION 3.10)

find_package(PythonInterp REQUIRED)

message(STATUS "Python: ${PYTHON_EXECUTABLE}")
message(STATUS "Bin Dir: ${CMAKE_BINARY_DIR}")
#
# Wheel file
#
set(MODEL_NAME "%s")
set(MODEL_LIBRARY "%s")
set(TRTIS_CIDMGR_WHEEL "trtis_cidmgr-${PROJECT_VERSION}-py2.py3-none-any.whl")

# copy files to build dir and build template files
file(COPY trtis_cidmgr DESTINATION .)
configure_file(../../config.pbtxt.in trtis_cidmgr/config.pbtxt.in)
configure_file(../../codes.in trtis_cidmgr/codes.py)
configure_file(version.py.in version.py)
configure_file(version.py.in trtis_cidmgr/version.py)
configure_file(setup.py setup.py COPYONLY)
configure_file(../../../README.md README.md COPYONLY)

file(GLOB_RECURSE TRTIS_CIDMGR_DEPS
    "*.py"
    "*.in"
    "../../*.in"
)

add_custom_command(
  OUTPUT "dist/${TRTIS_CIDMGR_WHEEL}"
  COMMAND "${PYTHON_EXECUTABLE}"
  ARGS
    "setup.py"
    "bdist_wheel"
    "--universal"
  DEPENDS
    ${TRTIS_CIDMGR_DEPS}
)

add_custom_target(
  client-wheel ALL
  DEPENDS
    "dist/${TRTIS_CIDMGR_WHEEL}"
)

set(ENV_BASE "install/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.env")
set(INSTALL_ENV "${CMAKE_BINARY_DIR}/${ENV_BASE}")
set(INSTALL_WHEELHOUSE "${CMAKE_BINARY_DIR}/install/wheelhouse")

install(
    FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/dist/${TRTIS_CIDMGR_WHEEL}"
    DESTINATION "${INSTALL_WHEELHOUSE}/."
)
install(
    DIRECTORY "${TRTUS_CLIENT_WHEELHOUSE}/"
    DESTINATION "${INSTALL_WHEELHOUSE}/"
    FILES_MATCHING PATTERN "*.whl"
)

install(
    CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m virtualenv ${INSTALL_ENV})"
    CODE "execute_process(COMMAND ${INSTALL_ENV}/bin/pip install --upgrade --force-reinstall -f ${INSTALL_WHEELHOUSE} trtis-cidmgr)"
    CODE "message(STATUS \"Wheels:     install/wheelhouse/${TRTIS_CIDMGR_WHEEL}\")"
    CODE "message(STATUS \"Virtualenv: ${ENV_BASE}\")"
)
