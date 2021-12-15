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

file(GLOB tests RELATIVE ${CMAKE_CURRENT_LIST_DIR} "t??-*")
foreach(test ${tests})
  add_subdirectory(${test})
endforeach()

include(ExternalProject)
string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
ExternalProject_Add(owl_as_subdirectory
  PREFIX owl_as_submodule
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/owl-as-subdirectory
  LIST_SEPARATOR | # Use the alternate list separator substituted above
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
  INSTALL_COMMAND ""
  BUILD_ALWAYS OFF
)
