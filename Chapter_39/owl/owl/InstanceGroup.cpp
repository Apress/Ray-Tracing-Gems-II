// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "InstanceGroup.h"
#include "Context.h"

#define LOG(message)                                    \
  if (Context::logging())                               \
    std::cout << "#owl(" << device->ID << "): "         \
              << message                                \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl(" << device->ID << "): "                 \
              << message << OWL_TERMINAL_DEFAULT << std::endl


namespace owl {

  /*! constructor */
  InstanceGroup::DeviceData::DeviceData(const DeviceContext::SP &device)
    : Group::DeviceData(device)
  {};

  InstanceGroup::InstanceGroup(Context *const context,
                               size_t numChildren,
                               Group::SP *groups,
                               unsigned int _buildFlags)
    : Group(context,context->groups),
      children(numChildren),
      buildFlags( (_buildFlags > 0) ? _buildFlags : defaultBuildFlags)
  {
    std::vector<uint32_t> childIDs;
    if (groups) {
      childIDs.resize(numChildren);
      for (size_t i=0;i<numChildren;i++) {
        assert(groups[i]);
        children[i] = groups[i];
        childIDs[i] = groups[i]->ID;
      }
    }

    transforms[0].resize(children.size());
    // do NOT automatically resize transforms[0] - need these only if
    // we use motion blur for this object
  }
  
  
  /*! pretty-printer, for printf-debugging */
  std::string InstanceGroup::toString() const
  {
    return "InstanceGroup";
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP InstanceGroup::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }
  
  /*! set transformation matrix of given child */
  void InstanceGroup::setTransform(size_t childID,
                                   const affine3f &xfm)
  {
    assert(childID < children.size());
    transforms[0][childID] = xfm;
  }

  void InstanceGroup::setTransforms(uint32_t timeStep,
                                    const float *floatsForThisStimeStep,
                                    OWLMatrixFormat matrixFormat)
  {
    switch(matrixFormat) {
    case OWL_MATRIX_FORMAT_OWL: {
      transforms[timeStep].resize(children.size());
      memcpy((char*)transforms[timeStep].data(),floatsForThisStimeStep,
             children.size()*sizeof(affine3f));
    } break;
    default:
      OWL_RAISE("used matrix format not yet implmeneted for"
                " InstanceGroup::setTransforms");
    };
  }

  /* set instance IDs to use for the children - MUST be an array of children.size() items */
  void InstanceGroup::setInstanceIDs(const uint32_t *_instanceIDs)
  {
    instanceIDs.resize(children.size());
    std::copy(_instanceIDs,_instanceIDs+instanceIDs.size(),instanceIDs.data());
  }

  /* set visibility masks to use for the children - MUST be an array of children.size() items */
  void InstanceGroup::setVisibilityMasks(const uint8_t *_visibilityMasks)
  {
    visibilityMasks.resize(children.size());
    std::copy(_visibilityMasks,_visibilityMasks+visibilityMasks.size(),visibilityMasks.data());
  }
  
  void InstanceGroup::setChild(size_t childID, Group::SP child)
  {
    assert(childID < children.size());
    children[childID] = child;
  }

  void InstanceGroup::buildAccel()
  {
    for (auto device : context->getDevices())
      if (transforms[1].empty())
        staticBuildOn<true>(device);
      else
        motionBlurBuildOn<true>(device);
  }
  
  void InstanceGroup::refitAccel()
  {
    for (auto device : context->getDevices())
      if (transforms[1].empty())
        staticBuildOn<false>(device);
      else
        motionBlurBuildOn<false>(device);
  }

  template<bool FULL_REBUILD>
  void InstanceGroup::staticBuildOn(const DeviceContext::SP &device) 
  {
    DeviceData &dd = getDD(device);
    auto optixContext = device->optixContext;

    SetActiveGPU forLifeTime(device);
    LOG("building instance accel over "
        << children.size() << " groups");

    // ==================================================================
    // sanity check that that many instances are actualy allowed by optix:
    // ==================================================================
    uint32_t maxInstsPerIAS = 0;
    optixDeviceContextGetProperty
      (optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
       &maxInstsPerIAS,
       sizeof(maxInstsPerIAS));
      
    if (children.size() > maxInstsPerIAS)
      throw std::runtime_error("number of children in instance group exceeds "
                               "OptiX's MAX_INSTANCES_PER_IAS limit");

    if (!FULL_REBUILD && !(buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE))
      throw std::runtime_error("trying to refit an accel struct that was not built with OPTIX_BUILD_FLAG_ALLOW_UPDATE");
    
    if (FULL_REBUILD) {
      dd.memFinal = 0;
      dd.memPeak = 0;
    }
   

    // ==================================================================
    // create instance build inputs
    // ==================================================================
    OptixBuildInput              instanceInput  {};
    OptixAccelBuildOptions       accelOptions   {};
    
    //! the N build inputs that go into the builder
    std::vector<OptixInstance>   optixInstances(children.size());

    // now go over all children to set up the buildinputs
    for (size_t childID=0;childID<children.size();childID++) {
      Group::SP child = children[childID];
      assert(child);

      assert(transforms[1].empty());
      const affine3f xfm = transforms[0][childID];

      OptixInstance oi = {};
      oi.transform[0*4+0]  = xfm.l.vx.x;
      oi.transform[0*4+1]  = xfm.l.vy.x;
      oi.transform[0*4+2]  = xfm.l.vz.x;
      oi.transform[0*4+3]  = xfm.p.x;
        
      oi.transform[1*4+0]  = xfm.l.vx.y;
      oi.transform[1*4+1]  = xfm.l.vy.y;
      oi.transform[1*4+2]  = xfm.l.vz.y;
      oi.transform[1*4+3]  = xfm.p.y;
        
      oi.transform[2*4+0]  = xfm.l.vx.z;
      oi.transform[2*4+1]  = xfm.l.vy.z;
      oi.transform[2*4+2]  = xfm.l.vz.z;
      oi.transform[2*4+3]  = xfm.p.z;
        
      oi.flags             = OPTIX_INSTANCE_FLAG_NONE;
      oi.instanceId        = (instanceIDs.empty())?uint32_t(childID):instanceIDs[childID];
      oi.visibilityMask    = (visibilityMasks.empty()) ? 255 : visibilityMasks[childID];
      oi.sbtOffset         = context->numRayTypes * child->getSBTOffset();
      oi.traversableHandle = child->getTraversable(device);
      assert(oi.traversableHandle);
      
      optixInstances[childID] = oi;
    }

    dd.optixInstanceBuffer.alloc(optixInstances.size()*
                                 sizeof(optixInstances[0]));
    dd.optixInstanceBuffer.upload(optixInstances.data(),"optixinstances");
    
    // ==================================================================
    // set up build input
    // ==================================================================
    instanceInput.type
      = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances
      = (CUdeviceptr)dd.optixInstanceBuffer.get();
    instanceInput.instanceArray.numInstances
      = (int)optixInstances.size();
      
    // ==================================================================
    // set up accel uptions
    // ==================================================================
    accelOptions.buildFlags = this->buildFlags;

    accelOptions.motionOptions.numKeys = 1;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
      
    // ==================================================================
    // query build buffer sizes, and allocate those buffers
    // ==================================================================
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                             &accelOptions,
                                             &instanceInput,
                                             1, // num build inputs
                                             &blasBufferSizes
                                             ));
    
    // ==================================================================
    // trigger the build ....
    // ==================================================================
    const size_t tempSize
      = FULL_REBUILD
      ? blasBufferSizes.tempSizeInBytes
      : blasBufferSizes.tempUpdateSizeInBytes;
    LOG("starting to build/refit "
        << prettyNumber(optixInstances.size()) << " instances, "
        << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
        << prettyNumber(tempSize) << "B in temp data");
      
    DeviceMemory tempBuffer;
    tempBuffer.alloc(tempSize);
      
    if (FULL_REBUILD) {
      dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
      dd.memPeak += tempBuffer.size();
      dd.memPeak += dd.bvhMemory.size();
      dd.memFinal = dd.bvhMemory.size();
    }
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* todo: stream */0,
                                &accelOptions,
                                // array of build inputs:
                                &instanceInput,1,
                                // buffer of temp memory:
                                (CUdeviceptr)tempBuffer.get(),
                                tempBuffer.size(),
                                // where we store initial, uncomp bvh:
                                (CUdeviceptr)dd.bvhMemory.get(),
                                dd.bvhMemory.size(),
                                /* the traversable we're building: */ 
                                &dd.traversable,
                                /* no compaction for instances: */
                                nullptr,0u
                                ));
      
    OWL_CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    // TODO: move those free's to the destructor, so we can delay the
    // frees until all objects are done
    tempBuffer.free();
      
    LOG_OK("successfully built instance group accel");
  }
    



  template<bool FULL_REBUILD>
  void InstanceGroup::motionBlurBuildOn(const DeviceContext::SP &device)
  {
    DeviceData &dd = getDD(device);
    auto optixContext = device->optixContext;
    
    SetActiveGPU forLifeTime(device);
    LOG("building instance accel over "
        << children.size() << " groups");
    
    // ==================================================================
    // sanity check that that many instances are actualy allowed by optix:
    // ==================================================================
    uint32_t maxInstsPerIAS = 0;
    optixDeviceContextGetProperty
      (optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
       &maxInstsPerIAS,
       sizeof(maxInstsPerIAS));
    
    if (children.size() > maxInstsPerIAS)
      throw std::runtime_error("number of children in instnace group exceeds "
                               "OptiX's MAX_INSTANCES_PER_IAS limit");
    
    // ==================================================================
    // build motion transforms
    // ==================================================================
    assert(!transforms[1].empty());
    std::vector<OptixMatrixMotionTransform> motionTransforms(children.size());
#if OPTIX_VERSION >= 70200
    /* since 7.2, optix no longer requires those aabbs (and in fact,
       no longer supports specifying them */
#else
    std::vector<box3f> motionAABBs(children.size());
#endif
    for (size_t childID=0;childID<children.size();childID++) {
      Group::SP child = children[childID];
      assert(child);
      OptixMatrixMotionTransform mt = {};
      mt.child                      = child->getTraversable(device);
      mt.motionOptions.numKeys      = 2;
      mt.motionOptions.timeBegin    = 0.f;
      mt.motionOptions.timeEnd      = 1.f;
      mt.motionOptions.flags        = OPTIX_MOTION_FLAG_NONE;

      for (int timeStep = 0; timeStep < 2; timeStep ++ ) {
        const affine3f xfm = transforms[timeStep][childID];
        mt.transform[timeStep][0*4+0]  = xfm.l.vx.x;
        mt.transform[timeStep][0*4+1]  = xfm.l.vy.x;
        mt.transform[timeStep][0*4+2]  = xfm.l.vz.x;
        mt.transform[timeStep][0*4+3]  = xfm.p.x;
          
        mt.transform[timeStep][1*4+0]  = xfm.l.vx.y;
        mt.transform[timeStep][1*4+1]  = xfm.l.vy.y;
        mt.transform[timeStep][1*4+2]  = xfm.l.vz.y;
        mt.transform[timeStep][1*4+3]  = xfm.p.y;
          
        mt.transform[timeStep][2*4+0]  = xfm.l.vx.z;
        mt.transform[timeStep][2*4+1]  = xfm.l.vy.z;
        mt.transform[timeStep][2*4+2]  = xfm.l.vz.z;
        mt.transform[timeStep][2*4+3]  = xfm.p.z;
      }

      motionTransforms[childID] = mt;

#if OPTIX_VERSION >= 70200
    /* since 7.2, optix no longer requires those aabbs (and in fact,
       no longer supports specifying them */
#else
      motionAABBs[childID]
        = xfmBounds(transforms[0][childID],child->bounds[0]);
      motionAABBs[childID].extend(xfmBounds(transforms[1][childID],child->bounds[1]));
#endif
    }
    // and upload
    dd.motionTransformsBuffer.alloc(motionTransforms.size()*
                                    sizeof(motionTransforms[0]));
    dd.motionTransformsBuffer.upload(motionTransforms.data(),"motionTransforms");
      
#if OPTIX_VERSION >= 70200
    /* since 7.2, optix no longer requires those aabbs (and in fact,
       no longer supports specifying them */
#else
    dd.motionAABBsBuffer.alloc(motionAABBs.size()*sizeof(box3f));
    dd.motionAABBsBuffer.upload(motionAABBs.data(),"motionaabbs");
#endif      
    // ==================================================================
    // create instance build inputs
    // ==================================================================
    OptixBuildInput              instanceInput  {};
    OptixAccelBuildOptions       accelOptions   {};
      
    //! the N build inputs that go into the builder
    std::vector<OptixInstance>   optixInstances(children.size());

    // now go over all children to set up the buildinputs
    for (size_t childID=0;childID<children.size();childID++) {
      Group::SP child = children[childID];
      assert(child);

      OptixTraversableHandle childMotionHandle = 0;
      OPTIX_CHECK(optixConvertPointerToTraversableHandle
                  (optixContext,
                   (CUdeviceptr)(((const uint8_t*)dd.motionTransformsBuffer.get())
                                 +childID*sizeof(motionTransforms[0])
                                 ),
                   OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                   &childMotionHandle));
        
      OptixInstance oi    = {};
      oi.transform[0*4+0]  = 1.f;//xfm.l.vx.x;
      oi.transform[0*4+1]  = 0.f;//xfm.l.vy.x;
      oi.transform[0*4+2]  = 0.f;//xfm.l.vz.x;
      oi.transform[0*4+3]  = 0.f;//xfm.p.x;
        
      oi.transform[1*4+0]  = 0.f;//xfm.l.vx.y;
      oi.transform[1*4+1]  = 1.f;//xfm.l.vy.y;
      oi.transform[1*4+2]  = 0.f;//xfm.l.vz.y;
      oi.transform[1*4+3]  = 0.f;//xfm.p.y;
        
      oi.transform[2*4+0]  = 0.f;//xfm.l.vx.z;
      oi.transform[2*4+1]  = 0.f;//xfm.l.vy.z;
      oi.transform[2*4+2]  = 1.f;//xfm.l.vz.z;
      oi.transform[2*4+3]  = 0.f;//xfm.p.z;
        
      oi.flags             = OPTIX_INSTANCE_FLAG_NONE;
      oi.instanceId        = (instanceIDs.empty())?uint32_t(childID):instanceIDs[childID];
      oi.sbtOffset         = context->numRayTypes * child->getSBTOffset();
      oi.visibilityMask    = (visibilityMasks.empty()) ? 255 : visibilityMasks[childID];
      oi.traversableHandle = childMotionHandle; 
      optixInstances[childID] = oi;
    }

    dd.optixInstanceBuffer.alloc(optixInstances.size()*
                                 sizeof(optixInstances[0]));
    dd.optixInstanceBuffer.upload(optixInstances.data(),"optixinstances");

    // ==================================================================
    // set up build input
    // ==================================================================
    instanceInput.type
      = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      
    instanceInput.instanceArray.instances
      = dd.optixInstanceBuffer.d_pointer;
    instanceInput.instanceArray.numInstances
      = (int)optixInstances.size();

#if OPTIX_VERSION >= 70200
    /* since 7.2, optix no longer requires those aabbs (and in fact,
       no longer supports specifying them */
#else
    instanceInput.instanceArray.aabbs
      = dd.motionAABBsBuffer.d_pointer;
    instanceInput.instanceArray.numAabbs
      = (int)motionAABBs.size();
#endif
    
      
    // ==================================================================
    // set up accel uption
    // ==================================================================
    accelOptions = {};
    accelOptions.buildFlags =
      OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
      |
      OPTIX_BUILD_FLAG_ALLOW_UPDATE
      ;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else {
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
    }
      
    // ==================================================================
    // query build buffer sizes, and allocate those buffers
    // ==================================================================
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                             &accelOptions,
                                             &instanceInput,
                                             1, // num build inputs
                                             &blasBufferSizes
                                             ));
    
    // ==================================================================
    // trigger the build ....
    // ==================================================================
    const size_t tempSize
      = FULL_REBUILD
      ? blasBufferSizes.tempSizeInBytes
      : blasBufferSizes.tempUpdateSizeInBytes;
    LOG("starting to build/refit "
        << prettyNumber(optixInstances.size()) << " instances, "
        << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
        << prettyNumber(tempSize) << "B in temp data");
      
    DeviceMemory tempBuffer;
    tempBuffer.alloc(tempSize);
      
    if (FULL_REBUILD)
      dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* todo: stream */0,
                                &accelOptions,
                                // array of build inputs:
                                &instanceInput,1,
                                // buffer of temp memory:
                                (CUdeviceptr)tempBuffer.get(),
                                tempBuffer.size(),
                                // where we store initial, uncomp bvh:
                                (CUdeviceptr)dd.bvhMemory.get(),
                                dd.bvhMemory.size(),
                                /* the traversable we're building: */ 
                                &dd.traversable,
                                /* no compaction for instances: */
                                nullptr,0u
                                ));

    OWL_CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    // TODO: move those free's to the destructor, so we can delay the
    // frees until all objects are done
    tempBuffer.free();
      
    LOG_OK("successfully built instance group accel");
  }
  
} // ::owl
