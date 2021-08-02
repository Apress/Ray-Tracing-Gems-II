if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
if (POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()
project(Optix-OWL VERSION 1.0.4)
cmake_minimum_required(VERSION 2.8)

if(WIN32)
  add_definitions(-DNOMINMAX)
endif()

set(owl_dir ${PROJECT_SOURCE_DIR}/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#include(configure_build_type)

add_definitions(${OWL_CXX_FLAGS})
include_directories(${OWL_INCLUDES})
include_directories(${OWL_VIEWER_INCLUDES})

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
find_program(BIN2C bin2c 
  HINTS 
  ${CUDA_TOOLKIT_ROOT_DIR}/bin/
  /usr/local/cuda/bin)

cuda_compile_and_embed(deviceCode_ptx deviceCode.cu)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${OWL_CXX_FLAGS}")
cuda_add_executable(rtxITS
  ${deviceCode_ptx}
  viewer.cpp
  renderer.cpp
)

set(LIBRARIES 
  ${OWL_LIBRARIES}
  ${OWL_VIEWER_LIBRARIES}
)

target_link_libraries(rtxITS ${LIBRARIES})

