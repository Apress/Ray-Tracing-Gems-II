# ======================================================================== #
# Copyright 2019-2020 Ingo Wald                                            #
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

# for STB image read/write
include_directories(${PROJECT_SOURCE_DIR}/samples/common/3rdParty)
# for public OWL API:
include_directories(${OWL_INCLUDES})

# ------------------------------------------------------------------
#cmdline-only samples, also used for 'make test'
# ------------------------------------------------------------------
add_subdirectory(cmdline)


# ------------------------------------------------------------------
# advanced sapmles and interactive samples require the interative
# viewer widget. try building this, and include those only if tihs
# worked (it may fail if liiraries like opengl, glfw, etc are missing)
# ------------------------------------------------------------------
add_subdirectory(common/owlViewer)

if (OWL_HAVE_VIEWER)
  
  # interactive tutorial-style samples (these will try to open windows)
  add_subdirectory(interactive)

  # advanced samples (these will try to open windows)
  add_subdirectory(advanced)

else()
  message(WARNING "#owl.cmake: could not build owlViewer library; skipping advanced samples")
endif()



# -------------------------------------------------------
# export those back to our parent
# -------------------------------------------------------
set(OWL_VIEWER_INCLUDES  ${OWL_VIEWER_INCLUDES}  PARENT_SCOPE)
set(OWL_VIEWER_LIBRARIES ${OWL_VIEWER_LIBRARIES} PARENT_SCOPE)
set(OWL_HAVE_TBB         ${OWL_HAVE_TBB}         PARENT_SCOPE)

