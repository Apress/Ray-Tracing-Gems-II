# ======================================================================== #
# Copyright 2019 Ingo Wald                                                 #
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
    test02-group-rebuilds-embedded-ptx
  PTX_TARGET
    test02-group-rebuilds-ptx
  SOURCES
    deviceCode.cu
)

target_link_libraries(test02-group-rebuilds-ptx PRIVATE owl::owl)

add_executable(test02-group-rebuilds hostCode.cpp)
target_link_libraries(test02-group-rebuilds
  PRIVATE
    test02-group-rebuilds-embedded-ptx
    owl::owl
    stb_image
)
add_test(test02-group-rebuilds ${CMAKE_BINARY_DIR}/test02-group-rebuilds)
