# Copyright (c) 2019 Doug Napoleone, All rights reserved.

cmake_minimum_required (VERSION 3.10)
project (cidmgr-client)

set(Protobuf_DIR ${TRTIS_BUILDDIR}/protobuf/lib/cmake/protobuf)
set(gRPC_DIR ${TRTIS_BUILDDIR}/grpc/lib/cmake/grpc)
set(c-ares_DIR ${TRTIS_BUILDDIR}/c-ares/lib/cmake/c-ares)

#
# Dependencies
#

include(FindOpenSSL)

# c-ares
find_package(c-ares CONFIG REQUIRED)

# Protobuf
set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

# GRPC
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

# Curl
find_package(CURL REQUIRED)


set(CODES_PREFIX "namespace dnapoleone { namespace inferenceserver { namespace correlation_id_mgr { namespace client { typedef enum cidmgr_code_enum {")
set(CODE_POSTFIX ",")
set(CODES_POSTFIX "} CIDMGR_Code; }}}}")
setshared(SEQUENCE_LIBRARY "sequence")
setshared(MODEL_LIBRARY "cidmgr")

configure_file(../../codes.in cidmgr_codes.h)

# set up the linking to the request client library.
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${TRTIS_CLIENT_INCLUDE}
)
link_directories(
  ${TRTIS_CLIENT_LIB}
)

#
# libcidmgr_client.a
#
add_library(
  cidmgr_client STATIC
  cidmgr_client.cc cidmgr_client.h cidmgr_codes.h
)
target_link_libraries(
  cidmgr_client
  PRIVATE request_static
  PRIVATE gRPC::grpc++
  PRIVATE gRPC::grpc
  PUBLIC protobuf::libprotobuf
  PUBLIC ${CURL_LIBRARY}
)

set(_LIB ${CMAKE_BINARY_DIR}/install/lib/)
set(_INCLUDE ${CMAKE_BINARY_DIR}/install/include/)
set(_BIN ${CMAKE_BINARY_DIR}/install/bin/)

install(
  TARGETS cidmgr_client
  LIBRARY DESTINATION ${_LIB}
)

install(
  FILES cidmgr_client.h
  DESTINATION ${_INCLUDE}
)

#
# cidmgr_sequence_client
#
add_executable(cidmgr_sequence_client cidmgr_sequence_client.cc)
target_link_libraries(
  cidmgr_sequence_client
  PRIVATE cidmgr_client
  PRIVATE request_static
  PRIVATE gRPC::grpc++
  PRIVATE gRPC::grpc
  PUBLIC protobuf::libprotobuf
  PUBLIC ${CURL_LIBRARY}
)
install(
  TARGETS cidmgr_sequence_client
  RUNTIME DESTINATION ${_BIN}
)
