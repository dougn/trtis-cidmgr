# Copyright (c) 2019 Doug Napoleone, All rights reserved.

cmake_minimum_required (VERSION 3.10)
project (cidmgr-client)


set(CODES_PREFIX "namespace dnapoleone { namespace inferenceserver { namespace correlation_id_mgr { namespace client { typedef enum cidmgr_code_enum {")
set(CODE_POSTFIX ",")
set(CODES_POSTFIX "} CIDMGR_Code; }}}}")
setshared(SEQUENCE_LIBRARY "sequence")
setshared(MODEL_LIBRARY "cidmgr")

configure_file(../../codes.in cidmgr_codes.h)

# set up the linking to the request client library.
##   OMG, trtis has botched the include paths for their client framework SOOOOO BAD!!!
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${TRTIS_BUILDDIR}/trtis-clients
  ${TRTIS_BUILDDIR}/..
  ${TRTIS_BUILDDIR}/protobuf/include/
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
  PRIVATE request
)

set(_LIB ${CMAKE_BINARY_DIR}/install/lib/)
set(_INCLUDE ${CMAKE_BINARY_DIR}/install/include/)
set(_BIN ${CMAKE_BINARY_DIR}/install/bin/)
setshared(REQUEST_LIB request ${TRTIS_CLIENT_LIB})

install(
  TARGETS cidmgr_client
  LIBRARY DESTINATION ${_LIB}
)

install(
  FILES ${REQUEST_LIB}
  DESTINATION ${_LIB}
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
  PRIVATE request
)
install(
  TARGETS cidmgr_sequence_client
  RUNTIME DESTINATION ${_BIN}
)