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

#include "RayGen.h"
#include "Context.h"

namespace owl {

  // ------------------------------------------------------------------
  // RayGenType::DeviceData
  // ------------------------------------------------------------------
  
  /*! constructor */
  RayGenType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : RegisteredObject::DeviceData(device)
  {}
  
  // ------------------------------------------------------------------
  // RayGenType
  // ------------------------------------------------------------------

  /*! constructor, with all the info to describe this type */
  RayGenType::RayGenType(Context *const context,
                         Module::SP module,
                         const std::string &progName,
                         size_t varStructSize,
                         const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->rayGenTypes,varStructSize,varDecls),
      module(module),
      progName(progName)
  {}
  
  /*! pretty-typecast into derived classes */
  std::string RayGenType::toString() const
  {
    return "RayGenType";
  }
    
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP RayGenType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

  // ------------------------------------------------------------------
  // RayGen::DeviceData
  // ------------------------------------------------------------------
  
  RayGen::DeviceData::DeviceData(const DeviceContext::SP &device,
                                 size_t dataSize)
    : RegisteredObject::DeviceData(device),
      rayGenRecordSize(OPTIX_SBT_RECORD_HEADER_SIZE
                       + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(dataSize))
  {
    SetActiveGPU forLifeTime(device);
    
    sbtRecordBuffer.alloc(rayGenRecordSize);
  }

  // ------------------------------------------------------------------
  // RayGen
  // ------------------------------------------------------------------
  
  /*! create new raygen of given type */
  RayGen::RayGen(Context *const context,
                 RayGenType::SP type) 
    : SBTObject(context,context->rayGens,type)
  {
    assert(context);
    assert(type);
    assert(type->module);
    assert(type->progName != "");
  }

  /*! clean up... */
  RayGen::~RayGen()
  {
    for (auto device : context->getDevices()) {
      SetActiveGPU forLifeTime(device);
      getDD(device).sbtRecordBuffer.free();
    }
  }

  /*! pretty-printer, for printf-debugging */
  std::string RayGen::toString() const
  {
    return "RayGen";
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP RayGen::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device,type->varStructSize);
  }

  /*! write the given SBT record, using the given device's
    corresponding device-side data represenataion */
  void RayGen::writeSBTRecord(uint8_t *const sbtRecord,
                              const DeviceContext::SP &device)
  {
    auto &dd = type->getDD(device);
    
    // first, compute pointer to record:
    uint8_t *const sbtRecordHeader = sbtRecord;
    uint8_t *const sbtRecordData   = sbtRecord+OPTIX_SBT_RECORD_HEADER_SIZE;

    // ------------------------------------------------------------------
    // pack record header with the corresponding hit group:
    // ------------------------------------------------------------------
    OPTIX_CALL(SbtRecordPackHeader(dd.pg,sbtRecordHeader));
    
    // ------------------------------------------------------------------
    // then, write the data for that record
    // ------------------------------------------------------------------
    writeVariables(sbtRecordData,device);
  }  

  /*! execute a *synchronous* launch of this raygen program, of given
    dimensions - this will wait for the program to complete */
  void RayGen::launch(const vec2i &dims)
  {
    launchAsync(dims,context->dummyLaunchParams);
    context->dummyLaunchParams->sync();
  }

  /*! *launch* this raygen prog with given launch params, but do NOT
    wait for completion - this means the SBT shuld NOT be changed or
    rebuild until a launchParams->sync() has been done */
  void RayGen::launchAsync(const vec2i &dims,
                           const LaunchParams::SP &lp)
  {
    assert("check valid launch dims" && dims.x > 0);
    assert("check valid launch dims" && dims.y > 0);
      
    assert(!deviceData.empty());
    for (int deviceID=0;deviceID<(int)deviceData.size();deviceID++) {
      DeviceContext::SP device = context->getDevice(deviceID);
      SetActiveGPU forLifeTime(device);
      
      RayGen::DeviceData       &rgDD = getDD(device);
      LaunchParams::DeviceData &lpDD = lp->getDD(device);
      
      lp->writeVariables(lpDD.hostMemory.data(),device);
      lpDD.deviceMemory.uploadAsync(lpDD.hostMemory.data(),lpDD.stream);

      auto &sbt = lpDD.sbt;

      // -------------------------------------------------------
      // set raygen part of SBT 
      // -------------------------------------------------------
      sbt.raygenRecord
        = (CUdeviceptr)rgDD.sbtRecordBuffer.d_pointer;
      assert(sbt.raygenRecord);

      // -------------------------------------------------------
      // set miss progs part of SBT 
      // -------------------------------------------------------
      assert("check miss records built" && device->sbt.missProgRecordCount != 0);
      sbt.missRecordBase
        = (CUdeviceptr)device->sbt.missProgRecordsBuffer.get();
      sbt.missRecordStrideInBytes
        = (uint32_t)device->sbt.missProgRecordSize;
      sbt.missRecordCount
        = (uint32_t)device->sbt.missProgRecordCount;
      
      // -------------------------------------------------------
      // set hit groups part of SBT 
      // -------------------------------------------------------
      assert("check hit records built" && device->sbt.hitGroupRecordCount != 0);
      sbt.hitgroupRecordBase
        = (CUdeviceptr)device->sbt.hitGroupRecordsBuffer.get();
      sbt.hitgroupRecordStrideInBytes
        = (uint32_t)device->sbt.hitGroupRecordSize;
      sbt.hitgroupRecordCount
        = (uint32_t)device->sbt.hitGroupRecordCount;
      
      OPTIX_CALL(Launch(device->pipeline,
                        lpDD.stream,
                        (CUdeviceptr)lpDD.deviceMemory.get(),
                        lpDD.deviceMemory.sizeInBytes,
                        &lpDD.sbt,
                        dims.x,dims.y,1
                        ));

      /* note we do NOT sync here ! */
    }
  }

} // ::owl

