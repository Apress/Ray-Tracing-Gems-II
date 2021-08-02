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

include_directories(${PROJECT_SOURCE_DIR}/)
# for STB image read/write
include_directories(${PROJECT_SOURCE_DIR}/samples/common/3rdParty)
# public API:
include_directories(${OWL_INCLUDES})

file(GLOB tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "t??-*")
foreach(test ${tests})
  add_subdirectory(${test})
endforeach()
