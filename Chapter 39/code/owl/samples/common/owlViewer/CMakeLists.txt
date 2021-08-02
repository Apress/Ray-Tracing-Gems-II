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

include(configure_glfw)

if (OWL_HAVE_GLFW)
  
  set(owl_viewer_sources
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

  # this is doing the same using OptiX
  add_library(owl_viewer STATIC ${owl_viewer_sources})
  
  target_link_libraries(owl_viewer
    ${OWL_GLFW_LIBRARIES}
    ${OWL_LIBRARIES}
    )

  set(OWL_VIEWER_LIBRARIES
    owl_viewer
    ${OWL_GLFW_LIBRARIES}
    ${OWL_LIBRARIES}
    )

  set(OWL_HAVE_VIEWER ON)
  set(OWL_VIEWER_INCLUDES
    ${OWL_GLFW_INCLUDES}
    ${owl_dir}/samples/common/
    )
  set(OWL_VIEWER_LIBRARIES
    owl_viewer
    ${OWL_LIBRARIES}
    ${OWL_GLFW_LIBRARIES}
    glfw
    )
else()
  set(OWL_HAVE_VIEWER OFF)
endif()

# export those to whoever included this:
set(OWL_HAVE_VIEWER ${OWL_HAVE_VIEWER} PARENT_SCOPE)
set(OWL_VIEWER_INCLUDES ${OWL_VIEWER_INCLUDES} PARENT_SCOPE)
set(OWL_VIEWER_LIBRARIES ${OWL_VIEWER_LIBRARIES} PARENT_SCOPE)

