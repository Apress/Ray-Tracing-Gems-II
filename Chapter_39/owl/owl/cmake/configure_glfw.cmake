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

# configures the glfw library

set(OWL_HAVE_GLFW OFF)

if (TARGET glfw)
#  message("seems glfw is already included/built by somebody else!?")
  set(OWL_HAVE_GLFW ON)
  set(OWL_GLFW_INCLUDES ${owl_dir}/samples/common/3rdParty/glfw)
  set(OWL_GLFW_LIBRARIES glfw ${OPENGL_LIBRARIES})
else ()
  set(OpenGL_GL_PREFERENCE "LEGACY")
  set(OpenGL_GL_PREFERENCE "LEGACY" PARENT_SCOPE)
  #set(OpenGL_GL_PREFERENCE "GLVND")
  #set(OpenGL_GL_PREFERENCE "GLVND" PARENT_SCOPE)
  find_package(OpenGL)

  if (OpenGL_FOUND)
    if (WIN32)
      add_subdirectory(${owl_dir}/samples/common/3rdParty/glfw source_dir)
      set(OWL_HAVE_GLFW ON)
      set(OWL_GLFW_INCLUDES ${owl_dir}/samples/common/3rdParty/glfw)
      set(OWL_GLFW_LIBRARIES glfw ${OPENGL_LIBRARIES})
else()
    find_package(glfw3 QUIET)
    
    if (${glfw3_FOUND})
#      message("#owl.cmake: found glfw3 package")
      include_directories(${glfw3_DIR})
      set(OWL_HAVE_GLFW ON)
      set(OWL_GLFW_LIBRARIES glfw ${OPENGL_LIBRARIES})
      set(OWL_GLFW_INCLUDES ${glfw3_INCLUDES})

    else()
 #     message("#owl.cmake: found OpenGL, but did NOT find glfw3 in system - building glfw from source")
      add_subdirectory(${owl_dir}/samples/common/3rdParty/glfw source_dir)
      set(OWL_HAVE_GLFW ON)
      set(OWL_GLFW_INCLUDES ${owl_dir}/samples/common/3rdParty/glfw)
      set(OWL_GLFW_LIBRARIES glfw ${OPENGL_LIBRARIES})
#      set(glfw3_DIR ${owl_dir}/samples/common/3rdParty/glfw)
    endif()
    endif()
  endif()
endif()

set(OWL_HAVE_GLFW ${OWL_HAVE_GLFW} PARENT_SCOPE)

