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

# support libraries
add_subdirectory(common)

# ------------------------------------------------------------------
#cmdline-only samples, also used for 'make test'
# ------------------------------------------------------------------
add_subdirectory(cmdline)

if (TARGET owl_viewer)

  # interactive tutorial-style samples (these will try to open windows)
  add_subdirectory(interactive)

  # advanced samples (these will try to open windows)
  add_subdirectory(advanced)

else()
  message(WARNING "#owl.cmake: could not build owl_viewer library; skipping advanced samples")
endif()
