cmake_minimum_required (VERSION 3.10)
project (cidmgr-backend)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
if(WIN32)
set(MODEL_LIBRARY "cidmgr.dll")
elseif(APPLE)
set(MODEL_LIBRARY "libcidmgr.dylib")
else()
set(MODEL_LIBRARY "libcidmgr.so")
endif()

configure_file(../codes.in cidmgr.h)
configure_file(../config.pbtxt.in config.pbtxt)

add_library(
  cidmgr SHARED
  cidmgr.cc cidmgr.h
)

target_link_libraries(
  cidmgr
  PRIVATE ${TRTIS_CUSTOM_BACKEND_LIB}
)

# threading on non-windows
if(NOT WIN32)
  target_link_libraries(
    cidmgr
    PUBLIC -lpthread
  )
endif()
target_include_directories(cidmgr PRIVATE ${TRTIS_CUSTOM_BACKEND_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/config.pbtxt"
    DESTINATION "${CMAKE_BINARY_DIR}/install/model/cidmgr/config.pbtxt"
)
install(
    TARGETS cidmgr
    DESTINATION "${CMAKE_BINARY_DIR}/install/model/cidmgr/1/"
)
install(
    TARGETS cidmgr
    LIBRARY DESTINATION "${CMAKE_BINARY_DIR}/install/lib/"
)




