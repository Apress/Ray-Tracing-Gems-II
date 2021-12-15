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

#if (WIN32)
#set(OPENGL_DEPENDENCY "")
#else()
#set(OpenGL_GL_PREFERENCE "LEGACY")
#find_package(OpenGL QUIET)
#set(OPENGL_DEPENDENCY OpenGL::OpenGL)
#endif()
#find_package(OpenGL QUIET)

set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED OPTIONAL_COMPONENTS OpenGL)

add_library(owl_viewer STATIC)

target_sources(owl_viewer PRIVATE
  # add header files, so visual studio will properly show them as part of the solution
  OWLViewer.h
  Camera.h
  InspectMode.h
  FlyMode.h

  # the actual source files
  OWLViewer.cpp
  Camera.cpp
  InspectMode.cpp
  FlyMode.cpp
)

if(TARGET OpenGL::OpenGL)
  target_link_libraries(owl_viewer PUBLIC OpenGL::OpenGL)
else()
  target_link_libraries(owl_viewer PUBLIC OpenGL::GL)
endif()

target_link_libraries(owl_viewer
  PUBLIC
    owl::owl
    glfw
    stb_image
)

target_include_directories(owl_viewer PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
