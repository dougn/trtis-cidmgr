# Copyright (c) 2019 Doug Napoleone, All rights reserved.

cmake_minimum_required (VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## helper functions where the proper name is needed for configure_file
function(setshared varname libname)
  set(_SEP "")
  if(ARGV2 AND (NOT ARGV2 MATCHES /$) AND (NOT ARGV2 MATCHES \\$))
    set(_SEP "/")
  endif()
  if(WIN32)
    set(${varname} "${ARGV2}${_SEP}${libname}.dll" PARENT_SCOPE)
  elseif(APPLE)
    set(${varname} "${ARGV2}${_SEP}lib${libname}.dylib" PARENT_SCOPE)
  else()
    set(${varname} "${ARGV2}${_SEP}lib${libname}.so" PARENT_SCOPE)
  endif()
endfunction()

function(setstatic varname libname)
  set(_SEP "")
  if(ARGV2 AND (NOT ARGV2 MATCHES /$) AND (NOT ARGV2 MATCHES \\$))
    set(_SEP "/")
  endif()
  if(WIN32)
    set(${varname} "${ARGV2}${_SEP}${libname}.lib" PARENT_SCOPE)
  else()
    set(${varname} "${ARGV2}${_SEP}lib${libname}.a" PARENT_SCOPE)
  endif()
endfunction()


add_subdirectory(backend)
add_subdirectory(clients)
