# ======================================================================== #
# Copyright 2019-2021 Ingo Wald                                            #
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

find_package(OptiX REQUIRED)

add_library(owl)

include(cmake/configure_tbb.cmake)

target_sources(owl PRIVATE
  # -------------------------------------------------------
  # infrastructure
  # -------------------------------------------------------
  include/owl/owl.h
  include/owl/owl_host.h
  include/owl/owl_device.h
  include/owl/common/arrayND/array2D.h
  include/owl/common/arrayND/array3D.h
  include/owl/common/math/AffineSpace.h
  include/owl/common/math/box.h
  include/owl/common/math/constants.h
  include/owl/common/math/fixedpoint.h
  include/owl/common/math/LinearSpace.h
  include/owl/common/math/Quaternion.h
  include/owl/common/math/random.h
  include/owl/common/math/vec/compare.h
  include/owl/common/math/vec/functors.h
  include/owl/common/math/vec/rotate.h
  include/owl/common/math/vec.h
  include/owl/common/owl-common.h
  include/owl/common/parallel/parallel_for.h
  include/owl/owl.h
  include/owl/owl_device.h
  include/owl/owl_device_buffer.h
  include/owl/owl_host.h
  helper/cuda.h
  helper/optix.h
  common.h

  cmake/configure_tbb.cmake
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
  Geometry.h
  Geometry.cpp
  Triangles.h
  Triangles.cu
  UserGeom.h
  UserGeom.cu

  # -------------------------------------------------------
  # accel structures
  # -------------------------------------------------------
  Group.h
  Group.cpp
  InstanceGroup.h
  InstanceGroup.cpp
  TrianglesGeomGroup.h
  TrianglesGeomGroup.cpp
  UserGeomGroup.h
  UserGeomGroup.cpp
)

set_target_properties(owl
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
)

target_link_libraries(owl PUBLIC OptiX::OptiX)
#target_link_libraries(owl PUBLIC OptiX)
if (OWL_HAVE_TBB AND TBB_FOUND)
  target_link_libraries(owl PUBLIC ${TBB_LIBRARIES})
  target_include_directories(owl PUBLIC ${TBB_INCLUDE_DIR})
  target_compile_definitions(owl PUBLIC -DOWL_HAVE_TBB=1)
endif()

# bind OWL to CUDA (includes and library)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17)
  # for new cmake version, we can rely on the working cuda detection
  # mechanism
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(owl PUBLIC CUDA::cuda_driver CUDA::cudart_static)
else()
  # for older version of cmake (where CUDAToolkit detection doesn't
  # properly work) we fall back to search paths for hardcoded library
  # names and header name; this isn't clean, but applies only for a
  # small set of deprecated cmake versions, so OK for now.
  target_link_libraries(owl PUBLIC cuda)
  find_path(cuda_h_dir NAMES cuda.h HINTS /usr/local/cuda/include)
  target_include_directories(owl PUBLIC ${cuda_h_dir})
endif()

target_include_directories(owl
  PUBLIC
    /usr/local/cuda/include/
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

if (WIN32)
  target_compile_definitions(owl PUBLIC -DNOMINMAX)
endif()

add_library(owl::owl ALIAS owl)
