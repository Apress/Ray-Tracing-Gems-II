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

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)

include_directories(${PROJECT_SOURCE_DIR}/owl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cuda_include_directories(${PROJECT_SOURCE_DIR}/owl/include)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})


add_definitions(${OWL_CXX_FLAGS})


set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD ON)
if (NOT WIN32)
set(CUDA_NVCC_FLAGS --compiler-options -fPIC)
endif()

include_directories(${PROJECT_SOURCE_DIR}/)
include_directories(${PROJECT_SOURCE_DIR}/owl/include)

cuda_include_directories(${PROJECT_SOURCE_DIR}/owl/include)

set(OWL_SOURCES

  # -------------------------------------------------------
  # infrastructure
  # -------------------------------------------------------
  include/owl/owl.h
  cmake/configure_tbb.cmake
  cmake/configure_owl.cmake
  cmake/configure_optix.cmake
  cmake/configure_glfw.cmake
  cmake/FindOptiX.cmake

  # -------------------------------------------------------
  # API layer
  # -------------------------------------------------------
  APIContext.h
  APIContext.cpp
  APIHandle.h
  APIHandle.cpp
  impl.cpp
  
  # -------------------------------------------------------
  # data handling
  # -------------------------------------------------------
  Buffer.h
  Buffer.cpp
  Texture.h
  Texture.cpp

  
  # -------------------------------------------------------
  # base classes
  # -------------------------------------------------------
  Object.h
  Object.cpp
  Module.h
  Module.cpp
  RegisteredObject.h
  RegisteredObject.cpp
  DeviceContext.h
  DeviceContext.cpp
  
  ObjectRegistry.h
  ObjectRegistry.cpp
  Context.h
  Context.cpp

  # -------------------------------------------------------
  # program types
  # -------------------------------------------------------
  SBTObject.h
  SBTObject.cpp
  RayGen.h
  RayGen.cpp
  LaunchParams.h
  LaunchParams.cpp
  MissProg.h
  MissProg.cpp
  Variable.h
  Variable.cpp
  
  # -------------------------------------------------------
  # geometries
  # -------------------------------------------------------
  Geometry.cpp
  Triangles.cu
  UserGeom.cu
  
  # -------------------------------------------------------
  # accel structures
  # -------------------------------------------------------
  Group.cpp
  InstanceGroup.cpp
  TrianglesGeomGroup.cpp
  UserGeomGroup.cpp
)

cuda_add_library(owl_static
  ${OWL_SOURCES}
  STATIC
  )


# Nate: 11/3/20: I get "fatal error C1041: cannot open program database" 
# on windows from this "owl" shared target being built in parallel against 
# the "owl_static" target, as for some reason they seem to try to 
# write to the same .pdb file simultaneously.

# cuda_add_library(owl
#   ${OWL_SOURCES}
#   )

# target_link_libraries(owl
#    ${CUDA_LIBRARIES}
#    ${CUDA_CUDA_LIBRARY}
#   )

target_link_libraries(owl_static
   ${CUDA_LIBRARIES}
   ${CUDA_CUDA_LIBRARY}
  )


#set_target_properties(owl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(owl_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_compile_options(owl_static PRIVATE -fPIC)

# -------------------------------------------------------
# export those back to our parent:
# -------------------------------------------------------
set(OWL_INCLUDES ${OWL_INCLUDES} PARENT_SCOPE)
set(OWL_LIBRARIES ${OWL_LIBRARIES} PARENT_SCOPE)
