# Copyright (c) 2019 Doug Napoleone, All rights reserved.

cmake_minimum_required (VERSION 3.10)

project (cidmgr VERSION 0.0.1 LANGUAGES CXX)

include(CMakeDependentOption)
include(ExternalProject)

set(TRTIS_BUILDDIR "" CACHE FILEPATH "Location of the tensorrt-inference-server project builddir directory")


if (TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX) 
  if (NOT EXISTS ${TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX})
    message(FATAL_ERROR "Could not find TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX: ${TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX}")
  endif()
else()
  if (NOT EXISTS ${TRTIS_BUILDDIR})
    message(FATAL_ERROR "Could not find TRTIS_BUILDDIR: ${TRTIS_BUILDDIR}")
  endif()
  set(TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX "${TRTIS_BUILDDIR}/trtis-custom-backends/install")
endif()

if (TRTIS_CLIENTS_INSTALL_PREFIX)
  if (NOT EXISTS ${TRTIS_CLIENTS_INSTALL_PREFIX})
    message(FATAL_ERROR "Could not find TRTIS_CLIENTS_INSTALL_PREFIX: ${TRTIS_CLIENTS_INSTALL_PREFIX}")
  endif()
else()
  set(TRTIS_CLIENTS_INSTALL_PREFIX "${TRTIS_BUILDDIR}/trtis-clients/install")
endif()

set(TRTIS_CUSTOM_BACKEND_INCLUDE "${TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX}/include")
set(TRTIS_CUSTOM_BACKEND_LIB "${TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX}/lib")


if(NOT EXISTS ${TRTIS_CUSTOM_BACKEND_INCLUDE})
  message(FATAL_ERROR "Could not find the trtis-custom-backend include directory: ${TRTIS_CUSTOM_BACKEND_INCLUDE}")
endif()
if(NOT EXISTS ${TRTIS_CUSTOM_BACKEND_LIB})
  message(FATAL_ERROR "Could not find the trtis-custom-backend library: ${TRTIS_CUSTOM_BACKEND_LIB}")
endif()

set(TRTIS_CLIENT_INCLUDE "${TRTIS_CLIENTS_INSTALL_PREFIX}/include")
set(TRTIS_CLIENT_LIB "${TRTIS_CLIENTS_INSTALL_PREFIX}/lib")

set(TRTUS_CLIENT_WHEELHOUSE "${TRTIS_CLIENTS_INSTALL_PREFIX}/python")

add_subdirectory(src)
