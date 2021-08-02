// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "UserGeomGroup.h"
#include "Context.h"

#define LOG(message)                                            \
  if (Context::logging())                                       \
    std::cout << "#owl(" << device->ID << "): "                 \
              << message                                        \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl(" << device->ID << "): "                 \
              << message << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {
  
  UserGeomGroup::UserGeomGroup(Context *const context,
                               size_t numChildren)
    : GeomGroup(context,numChildren)
  {}

  void UserGeomGroup::buildOrRefit(bool FULL_REBUILD)
  {
    for (auto child : geometries) {
      UserGeom::SP userGeom = child->as<UserGeom>();
      assert(userGeom);
      for (auto device : context->getDevices())
        userGeom->executeBoundsProgOnPrimitives(device);
    }
    
    for (auto device : context->getDevices())
      if (FULL_REBUILD)
        buildAccelOn<true>(device);
      else
        buildAccelOn<false>(device);
  }
  
  void UserGeomGroup::buildAccel()
  {
    buildOrRefit(true);
  }

  void UserGeomGroup::refitAccel()
  {
    buildOrRefit(false);
  }

  /*! low-level accel structure builder for given device */
  template<bool FULL_REBUILD>
  void UserGeomGroup::buildAccelOn(const DeviceContext::SP &device)
  {
    DeviceData &dd = getDD(device);
    auto optixContext = device->optixContext;

    if (FULL_REBUILD && !dd.bvhMemory.empty())
      dd.bvhMemory.free();

    if (!FULL_REBUILD && dd.bvhMemory.empty())
      throw std::runtime_error("trying to refit an accel struct that has not been previously built");
    // // assert("check does not yet exist" && dd.traversable == 0);
    // if (FULL_REBUILD)
    //   assert("check does not yet exist on first build " && dd.bvhMemory.empty());
    // else
    //   assert("check DOES exist on first build " && !dd.bvhMemory.empty());
      
    SetActiveGPU forLifeTime(device);
    LOG("building user accel over "
        << geometries.size() << " geometries");

    size_t sumPrims = 0;
    uint32_t maxPrimsPerGAS = 0;
    optixDeviceContextGetProperty
      (device->optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
       &maxPrimsPerGAS,
       sizeof(maxPrimsPerGAS));
    
    // ==================================================================
    // create triangle inputs
    // ==================================================================
    //! the N build inputs that go into the builder
    std::vector<OptixBuildInput> userGeomInputs(geometries.size());
    /*! *arrays* of the vertex pointers - the buildinputs cointina
     *pointers* to the pointers, so need a temp copy here */
    std::vector<CUdeviceptr> boundsPointers(geometries.size());

    // for now we use the same flags for all geoms
    std::vector<uint32_t> userGeomInputFlags(geometries.size());

    // now go over all geometries to set up the buildinputs
    for (size_t childID=0;childID<geometries.size();childID++) {
      // the three fields we're setting:

      UserGeom::SP child = geometries[childID]->as<UserGeom>();
      assert(child);

      sumPrims += child->primCount;
      if (sumPrims > maxPrimsPerGAS) 
        throw std::runtime_error("number of prim in user geom group exceeds "
                                 "OptiX's MAX_PRIMITIVES_PER_GAS limit");

      UserGeom::DeviceData &ugDD = child->getDD(device);
      
      CUdeviceptr     &d_bounds = boundsPointers[childID];
      OptixBuildInput &userGeomInput = userGeomInputs[childID];

      assert("user geom has valid bounds buffer"
             && ugDD.internalBufferForBoundsProgram.alloced());
      d_bounds = (CUdeviceptr)ugDD.internalBufferForBoundsProgram.get();
      
      userGeomInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#if OPTIX_VERSION >= 70100
      auto &aa = userGeomInput.customPrimitiveArray;
#else
      auto &aa = userGeomInput.aabbArray;
#endif
      aa.aabbBuffers   = &d_bounds;
      aa.numPrimitives = (uint32_t)child->primCount;
      aa.strideInBytes = sizeof(box3f);
      aa.primitiveIndexOffset = 0;
      
      // we always have exactly one SBT entry per shape (i.e., triangle
      // mesh), and no per-primitive materials:
      userGeomInputFlags[childID]    = 0;
      aa.flags                       = &userGeomInputFlags[childID];
      // iw, jan 7, 2020: note this is not the "actual" number of
      // SBT entires we'll generate when we build the SBT, only the
      // number of per-ray-type 'groups' of SBT enties (i.e., before
      // scaling by the SBT_STRIDE that gets passed to
      // optixTrace. So, for the build input this value remains *1*).
      aa.numSbtRecords               = 1; 
      aa.sbtIndexOffsetBuffer        = 0; 
      aa.sbtIndexOffsetSizeInBytes   = 0; 
      aa.sbtIndexOffsetStrideInBytes = 0; 
    }

    // ==================================================================
    // BLAS setup: buildinputs set up, build the blas
    // ==================================================================
      
    // ------------------------------------------------------------------
    // first: compute temp memory for bvh
    // ------------------------------------------------------------------
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             =
      // OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
      OPTIX_BUILD_FLAG_ALLOW_UPDATE
      |
      OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
      ;
    accelOptions.motionOptions.numKeys  = 1;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 userGeomInputs.data(),
                 (uint32_t)userGeomInputs.size(),
                 &blasBufferSizes
                 ));
    
    // ------------------------------------------------------------------
    // ... and allocate buffers: temp buffer, initial (uncompacted)
    // BVH buffer, and a one-single-size_t buffer to store the
    // compacted size in
    // ------------------------------------------------------------------
      
    // temp memory:
    DeviceMemory tempBuffer;
    tempBuffer.alloc
      (FULL_REBUILD
       ? blasBufferSizes.tempSizeInBytes
       : blasBufferSizes.tempUpdateSizeInBytes);

    if (FULL_REBUILD)
      // alloc only on first rebuild
      dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* todo: stream */0,
                                &accelOptions,
                                // array of build inputs:
                                userGeomInputs.data(),
                                (uint32_t)userGeomInputs.size(),
                                // buffer of temp memory:
                                (CUdeviceptr)tempBuffer.get(),
                                tempBuffer.size(),
                                // where we store initial, uncomp bvh:
                                (CUdeviceptr)dd.bvhMemory.get(),
                                dd.bvhMemory.size(),
                                /* the dd.traversable we're building: */ 
                                &dd.traversable,
                                /* we're also querying compacted size: */
                                nullptr,0u
                                ));
      
    CUDA_SYNC_CHECK();

    // ==================================================================
    // finish - clean up
    // ==================================================================

    tempBuffer.free();

    LOG_OK("successfully built user geom group accel");

    // size_t sumPrims = 0;
    size_t sumBoundsMem = 0;
    for (size_t childID=0;childID<geometries.size();childID++) {
      UserGeom::SP child = geometries[childID]->as<UserGeom>();
      assert(child);
      
      UserGeom::DeviceData &ugDD = child->getDD(device);
      
      sumBoundsMem += ugDD.internalBufferForBoundsProgram.sizeInBytes;
      if (ugDD.internalBufferForBoundsProgram.alloced())
        ugDD.internalBufferForBoundsProgram.free();
    }

    CUDA_SYNC_CHECK();
  }
    
} // ::owl
