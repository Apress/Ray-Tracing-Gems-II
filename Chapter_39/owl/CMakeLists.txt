# ======================================================================== #
# Copyright 2019-2020 Ingo Wald                                            #
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

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(Optix-OWL VERSION 1.0.5)
cmake_minimum_required(VERSION 2.8)

OPTION(OWL_BUILD_SAMPLES "Build the Samples?" ON)
OPTION(OWL_BUILD_ADVANCED_TESTS "Build the *advanced* test-cases?" OFF)
OPTION(OWL_USE_TBB "Use TBB to parallelize computation?" ON)
if (NOT (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_HOME_DIRECTORY}))
  SET(OWL_IS_SUBPROJECT ON)
else()
  SET(OWL_IS_SUBPROJECT OFF)
endif()

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(owl_dir ${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
include(configure_owl)
include(configure_tbb)
# if not a subproject, set "Debug/Release flags and binary output dir
# to <binaryDir>
if (NOT OWL_IS_SUBPROJECT)
include(configure_build_type)
endif()

# iw, 1/3/20: yes, i know this is 'old' cmake style, but on CentOS7
# this is the only way that works, since the 'correct' way of setting
# the cxx11 standard doesnt propagate those flags to nvcc for the cuda
# programs. TODO: check to see if this can be fixed in the
# configure_cuda script instead.

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OWL_CXX_FLAGS "")
if (NOT WIN32)
  set (CMAKE_CXX_FLAGS "--std=c++11")
  set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()
if (WIN32)
  # Nate: 11/3/20: I get "fatal error C1041: cannot open program database" 
  # on windows from multiple targets writing to the same .pdb file simultaneously.
  # This flag doesn't seem to fix it.
  # set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/FS")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(configure_cuda)
include(configure_optix)
include(configure_owl)
include(configure_tbb)

include_directories(${OWL_INCLUDES})
add_definitions(${OWL_CXX_FLAGS})

# ------------------------------------------------------------------
# owl library itself, including owl-common
# ------------------------------------------------------------------
add_subdirectory(owl)

# ------------------------------------------------------------------
# tutorial/samples
# ------------------------------------------------------------------
if (OWL_BUILD_SAMPLES)
  enable_testing()
  add_subdirectory(samples)
endif()

# ------------------------------------------------------------------
# some basic testing
# ------------------------------------------------------------------
if (NOT OWL_IS_SUBPROJECT)
  if (OWL_BUILD_ADVANCED_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()
endif()

# ------------------------------------------------------------------
# check if this is included as a submodule, and if so, export some
# variables to the parent
# ------------------------------------------------------------------
if (OWL_IS_SUBPROJECT)
  set(OWL_INCLUDES ${OWL_INCLUDES} PARENT_SCOPE)
  set(OWL_VIEWER_INCLUDES ${OWL_VIEWER_INCLUDES} PARENT_SCOPE)
  set(OWL_LIBRARIES ${OWL_LIBRARIES} PARENT_SCOPE)
  set(OWL_VIEWER_LIBRARIES ${OWL_VIEWER_LIBRARIES} PARENT_SCOPE)
  set(OWL_HAVE_TBB ${OWL_HAVE_TBB} PARENT_SCOPE)
  set(OWL_CXX_FLAGS ${OWL_CXX_FLAGS} PARENT_SCOPE)
endif()
