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

#pragma once

#include "SBTObject.h"
#include "Module.h"

namespace owl {

  /*! describes a given *type* of launch params - basically the set of
      variables in the device-side '__global__ <Struct>
      optixLaumchParams' variable. This class describes the types and
      memory layouts of vairables in this 'Struct', the acutal
      instances of this type then store the variable values to be
      written into that variable */
  struct LaunchParamsType : public SBTObjectType {
    typedef std::shared_ptr<LaunchParamsType> SP;

    /*! constructor, with given set of variables */
    LaunchParamsType(Context *const context,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    virtual std::string toString() const { return "LaunchParamsType"; }
  };

  /*! an object that stores the variables used for building the launch
      params data - this is all this object does: store values and
      write them when requested */
  struct LaunchParams : public SBTObject<LaunchParamsType> {
    typedef std::shared_ptr<LaunchParams> SP;

    /*! device-specific data for these lauch params - each instance
        needs its own host- and device-side memory to store the
        parameter values (to avoid messing with other launches if and
        when we use multiple async launches in parallel!) */
    struct DeviceData : public RegisteredObject::DeviceData {

      /*! constructor, which allocs all the device-side data */
      DeviceData(const DeviceContext::SP &device, size_t  dataSize);
      
      const size_t            dataSize;
      
      OptixShaderBindingTable sbt = {};

      /*! host-size memory for the launch paramters - we have a
          host-side copy, too, so we can leave the launch2D call
          without having to first wait for the cudaMemcpy to
          complete */
      std::vector<uint8_t> hostMemory;
      
      /*! the cuda device memory we copy the launch params to */
      DeviceMemory         deviceMemory;
      
      /*! a cuda stream we can use for the async upload and the
          following async launch */
      cudaStream_t         stream = nullptr;
    };

    /*! create a new instenace of given launch param type */
    LaunchParams(Context *const context,
                 LaunchParamsType::SP type);
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! returns the cuda stream associated with this launch params
        object (for given device, since each device has a different
        one. Note this stream is different from the default optix
        context stream to allow asynchronous use of launhc params -
        ie, each set of launchparams has its oww stream and can thus
        be used/launched independently of other launchparam-based
        launches */
    CUstream getCudaStream(const DeviceContext::SP &device);

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
      
    /*! wait for the latest launch done with these launch params to
      complete, by syncing on the stream associated with these
      params */
    void sync();    
  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline LaunchParams::DeviceData &LaunchParams::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<LaunchParams::DeviceData>();
  }
  
} // ::owl

