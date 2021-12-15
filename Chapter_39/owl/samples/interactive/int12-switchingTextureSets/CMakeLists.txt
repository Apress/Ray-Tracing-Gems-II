# ======================================================================== #
# Copyright 2020 Ingo Wald                                                 #
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

# main purpose of this example is to showcase and test the sitation of
# having buffers of buffers of textures: Though primarily built upon
# the previous int11-rotatoinBoxes sample, in this sample we no longer
# have one randomly created texture per box, but instead, have N sets
# of 6 textures (one per face) of each box, using the current frame
# time to cycle through those texture sets. Ie, each box stores one
# one buffer of N buffers of 6 textures each

embed_ptx(
  OUTPUT_TARGET
    int12-switchingTextureSets-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

add_executable(int12-switchingTextureSets hostCode.cpp)
target_link_libraries(int12-switchingTextureSets
  PRIVATE
    int12-switchingTextureSets-ptx
    owl_viewer
)
