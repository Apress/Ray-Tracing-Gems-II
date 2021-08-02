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

cuda_compile_and_embed(ptxCode
  deviceCode.cu
  )
add_executable(int12-switchingTextureSets
  hostCode.cpp
  ${ptxCode}
  )
target_link_libraries(int12-switchingTextureSets
  ${OWL_LIBRARIES}
  ${OWL_VIEWER_LIBRARIES}
  )


