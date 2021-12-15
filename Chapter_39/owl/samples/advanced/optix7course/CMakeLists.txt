# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
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

embed_ptx(
  OUTPUT_TARGET
    adv_optix7course-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    devicePrograms.cu
)

add_executable(adv_optix7course
  # the file with embedded PTX string for the device programs:
  SampleRenderer.h
  SampleRenderer.cpp
  Model.h
  Model.cpp
  LaunchParams.h
  main.cpp
)

target_link_libraries(adv_optix7course PRIVATE adv_optix7course-ptx owl_viewer)
