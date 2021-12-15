# ======================================================================== #
# Copyright 2021-2021 Ingo Wald                                            #
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

add_subdirectory(stb_image)

if (OWL_BUILD_SAMPLES)
  find_package(glfw3 QUIET)
  set(OpenGL_GL_PREFERENCE "LEGACY")
  #find_package(OpenGL QUIET)
  find_package(OpenGL REQUIRED OPTIONAL_COMPONENTS OpenGL)
  if (NOT TARGET glfw)
    if (OpenGL_FOUND)
      message(STATUS "found opengl, building glfw")
      add_subdirectory(glfw)
    endif()
  endif()
endif()
