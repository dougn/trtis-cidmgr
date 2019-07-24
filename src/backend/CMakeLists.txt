# Copyright (c) 2019 Doug Napoleone, All rights reserved.

cmake_minimum_required (VERSION 3.10)
project (cidmgr-backend)

#
# libcidmgr.so
#

## tensorrt-inference-server is NOT properly set up for this.
## instead we should pull from the builddir for now.
## for docker we want to pull from system
## but the tensorrt-inference-server project does not set
## that up properly either.
##find_package(trtis-custom-backends REQUIRED) 


set(CODES_PREFIX "namespace dnapoleone { namespace inferenceserver { namespace correlation_id_mgr { namespace backend { typedef enum cidmgr_code_enum {")
set(CODE_POSTFIX ",")
set(CODES_POSTFIX "} CIDMGR_Code; }}}}")
set(MODEL_NAME "cidmgr")
setshared(MODEL_LIBRARY "cidmgr")
setshared(SEQUENCE_LIBRARY "sequence")

configure_file(../codes.in cidmgr.h)
configure_file(../config.pbtxt.in config.pbtxt)
configure_file("${CMAKE_SOURCE_DIR}/test/simple_sequence_config.pbtxt.in" sequence/config.pbtxt)

add_library(
  cidmgr SHARED
  cidmgr.cc cidmgr.h
)
setstatic(CUSTOMBACKEND "custombackend" "${TRTIS_CUSTOM_BACKEND_LIB}")

target_link_libraries(
  cidmgr
  PRIVATE ${CUSTOMBACKEND}
)

# threading on non-windows
if(NOT WIN32)
  target_link_libraries(
    cidmgr
    PUBLIC -lpthread
  )
  if(NOT APPLE)
    configure_file(libcidmgr.ldscript libcidmgr.ldscript COPYONLY)
    set_target_properties(
      cidmgr
      PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libcidmgr.ldscript
    )
    set_target_properties(
      cidmgr
      PROPERTIES LINK_FLAGS "-Wl,--version-script libcidmgr.ldscript"
    )
  endif()
endif()


target_include_directories(cidmgr PRIVATE ${TRTIS_CUSTOM_BACKEND_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})

# install/lib
set(_LIB ${CMAKE_BINARY_DIR}/install/lib/)
install(
  TARGETS cidmgr
  LIBRARY DESTINATION "${_LIB}"
)
install(
  FILES "${TRTIS_CUSTOM_BACKEND_LIB}/${SEQUENCE_LIBRARY}"
  DESTINATION "${_LIB}"
)

# install/model_repository
set(_MODEL_REPOSITORY "${CMAKE_BINARY_DIR}/install/model_repository")
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/config.pbtxt"
  DESTINATION "${_MODEL_REPOSITORY}/cidmgr/"
)
install(
  TARGETS cidmgr
  DESTINATION "${_MODEL_REPOSITORY}/cidmgr/1/"
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/sequence/config.pbtxt"
  DESTINATION "${_MODEL_REPOSITORY}/simple_sequence/"
)
install(
  FILES "${TRTIS_CUSTOM_BACKEND_LIB}/${SEQUENCE_LIBRARY}"
  DESTINATION "${_MODEL_REPOSITORY}/simple_sequence/1/"
)
