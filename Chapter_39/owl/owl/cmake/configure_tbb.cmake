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

OPTION(OWL_DISABLE_TBB "DISABLE TBB in OWL, even if it could be found" OFF)
if (OWL_DISABLE_TBB)
  set(OWL_USE_TBB OFF)
else()
  OPTION(OWL_USE_TBB "Use TBB to parallelize host-side code?" ON)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (OWL_USE_TBB AND (NOT OWL_DISABLE_TBB))
  find_package(TBB)

  if (TBB_FOUND)
    #    include_directories(${TBB_INCLUDE_DIR})
#    set(OWL_CXX_FLAGS "${OWL_CXX_FLAGS} -DOWL_HAVE_TBB=1")
    message(STATUS "#owl.cmake: found TBB, in include dir ${TBB_INCLUDE_DIR}")
    set(OWL_HAVE_TBB ON)
  else()
    message(STATUS "#owl.cmake: TBB not found; falling back to serial execution of owl::parallel_for")
    set(OWL_HAVE_TBB OFF)
  endif()
else()
  set(OWL_HAVE_TBB OFF)
endif()