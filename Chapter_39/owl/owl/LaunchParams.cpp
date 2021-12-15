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

#include "LaunchParams.h"
#include "Context.h"

namespace owl {

  // ------------------------------------------------------------------
  // LaunchParamsType
  // ------------------------------------------------------------------
  
  LaunchParamsType::LaunchParamsType(Context *const context,
                                     size_t varStructSize,
                                     const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->launchParamTypes,varStructSize,varDecls)
  {
  }
  
  // ------------------------------------------------------------------
  // LaunchParams::DeviceData
  // ------------------------------------------------------------------
  
  /*! constructor, which allocs all the device-side data */
  LaunchParams::DeviceData::DeviceData(const DeviceContext::SP &device,
                                       size_t  dataSize)
    : RegisteredObject::DeviceData(device),
      dataSize(dataSize),
      sbt({})
  {
    SetActiveGPU forLifeTime(device);
    
    OWL_CUDA_CHECK(cudaStreamCreate(&stream));
    deviceMemory.alloc(dataSize);
    hostMemory.resize(dataSize);
  }

  LaunchParams::DeviceData::~DeviceData()
  {
    cudaStreamDestroy(stream);
  }
  
  // ------------------------------------------------------------------
  // LaunchParams
  // ------------------------------------------------------------------
  
  LaunchParams::LaunchParams(Context *const context,
                             LaunchParamsType::SP type) 
    : SBTObject(context,context->launchParams,type)
  {
    assert(context);
    assert(type);
    assert(type.get());
  }

  /*! pretty-printer, for printf-debugging */
  std::string LaunchParams::toString() const 
  {
    return "LaunchParams";
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  LaunchParams::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device,type->varStructSize);
  }

  /*! returns the cuda stream associated with this launch params
    object (for given device, since each device has a different
    one. Note this stream is different from the default optix
    context stream to allow asynchronous use of launhc params -
    ie, each set of launchparams has its oww stream and can thus
    be used/launched independently of other launchparam-based
    launches */
  CUstream LaunchParams::getCudaStream(const DeviceContext::SP &device)
  {
    return getDD(device).stream;
  }

  /*! wait for the latest launch done with these launch params to
      complete, by syncing on the stream associated with these
      params */
  void LaunchParams::sync()
  {
    for (auto device : context->getDevices()) {
      SetActiveGPU forLifeTime(device);
      cudaStreamSynchronize(getCudaStream(device));
    }
  }
  
} // ::owl

