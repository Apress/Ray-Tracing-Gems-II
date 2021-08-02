# ======================================================================== #
# Copyright 2018-2020 Ingo Wald                                            #
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

OPTION(OWL_USE_TBB "Use TBB to parallelize host-side code?" ON)

if (OWL_USE_TBB)
  find_package(TBB)

  set(OWL_HAVE_TBB OFF)
  if (TBB_FOUND)
    #    include_directories(${TBB_INCLUDE_DIR})
    set(OWL_CXX_FLAGS "${OWL_CXX_FLAGS} -DOWL_HAVE_TBB=1")
    set(OWL_HAVE_TBB ON)
    
    if (TBB_FOUND)
      set(OWL_INCLUDES ${OWL_INCLUDES} ${TBB_INCLUDE_DIR})
      set(OWL_LIBRARIES ${OWL_LIBRARIES} ${TBB_LIBRARIES})
    endif()
  else()
    message("#owl.cmake: TBB not found; falling back to serial execution of owl::parallel_for")
  endif()
endif()