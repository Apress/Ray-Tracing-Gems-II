# ======================================================================== #
# Copyright 2019-2021 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(OWL_IS_SUBPROJECT ON)
else()
  set(OWL_IS_SUBPROJECT OFF)
endif()

if(CUDA_TOOLKIT_ROOT_DIR)
  # Somebody called find_cuda--they are trying to use the deprecated build.
  message(WARNING "CUDA_TOOLKIT_ROOT_DIR is set. "
    "This usually means that a parent project has called find_cuda, which "
    "has been deprecated in CMake since 2018 and is no longer supported by OWL. "
    "Instead, the parent project is required to use 'intermediate' or 'modern' "
    "CMake CUDA style with first-class CUDA language support (i.e., "
    "project(foo LANGUAGES C CXX CUDA)). "
  )
endif()

set(CMAKE_BUILD_TYPE_INIT "Release")

# project command is required to come after cmake_minimum_required command.
project(Optix-OWL VERSION 1.1.5 LANGUAGES C CXX CUDA)

#set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT OWL_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/owl/cmake/")
if (OWL_IS_SUBPROJECT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)
endif()
#include(configure_tbb)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(embed_ptx)

# ------------------------------------------------------------------
# owl library itself, including owl-common
# ------------------------------------------------------------------
option(OWL_BUILD_SHARED "Build OWL as a shared library? (otherwise static)" OFF)
set(BUILD_SHARED_LIBS ${OWL_BUILD_SHARED}) # use 'OWL_' naming convention
add_subdirectory(owl)

option(OWL_BUILD_SAMPLES "Build the Samples?" ON)
add_compile_definitions(OWL_BUILDING_ALL_SAMPLES)
option(OWL_BUILD_ADVANCED_TESTS "Build the *advanced* test-cases?" OFF)

add_subdirectory(3rdParty)

# ------------------------------------------------------------------
# tutorial/samples
# ------------------------------------------------------------------
if (OWL_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if (OWL_IS_SUBPROJECT)
  return()
endif()

include(CTest)
if (BUILD_TESTING)
  enable_testing()
endif()
# ------------------------------------------------------------------
# some basic testing
# ------------------------------------------------------------------
if (OWL_BUILD_ADVANCED_TESTS)
  add_subdirectory(tests)
endif()
