cmake_minimum_required(VERSION 3.17)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

project(RTX_ITS VERSION 1.0.4 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(owl EXCLUDE_FROM_ALL)

include(embed_ptx)

#include(configure_build_type)

embed_ptx(
  OUTPUT_TARGET
    deviceCode_ptx
  EMBEDDED_SYMBOL_NAMES
    deviceCode_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

add_executable(rtxITS
  viewer.cpp
  Renderer.cpp
)

target_link_libraries(rtxITS PRIVATE owl::owl owl_viewer deviceCode_ptx)
