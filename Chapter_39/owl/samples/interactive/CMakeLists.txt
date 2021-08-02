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

# ---------------------------------------------------------------------------
# a more serious kind of example with model loader, viewer, simple
# path tracer, etc
# ---------------------------------------------------------------------------
if (OWL_HAVE_VIEWER)
  
  file(GLOB samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "int??-*")
  
  include_directories(${OWL_VIEWER_INCLUDES})
  add_definitions(${OWL_CXX_FLAGS})
  foreach(sample ${samples})
    add_subdirectory(${sample})
  endforeach()
  
endif()


