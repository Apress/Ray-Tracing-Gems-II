## ======================================================================== ##
## Copyright 2018-2019 Ingo Wald                                            ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
#find_package(CUDA REQUIRED)
find_package(CUDA)

if (CUDA_FOUND)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()