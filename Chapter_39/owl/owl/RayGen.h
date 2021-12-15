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

#pragma once

#include "SBTObject.h"
#include "Module.h"
#include "LaunchParams.h"

namespace owl {

  /*! type that describes a raygen program's variables and programs */
  struct RayGenType : public SBTObjectType {
    typedef std::shared_ptr<RayGenType> SP;

    /*! describes the device-speific data for this program - optix has
        a differnet context per GPU, so also a different
        optixprogramgroup per GPU */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;
      
      /*! constructor */
      DeviceData(const DeviceContext::SP &device);
      
      /*! the optix-compiled program group witin the given device's
        optix context */
      OptixProgramGroup pg = 0;
    };
    
    /*! constructor, with all the info to describe this type */
    RayGenType(Context *const context,
               Module::SP module,
               const std::string &progName,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);
    
    /*! pretty-typecast into derived classes */
    std::string toString() const override;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
    
    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! the module in which the program is defined */
    Module::SP        module;
    
    /*! name of the program within this module */
    const std::string progName;
    /*! the name, annotated wiht optix' "__raygen__" */
    const std::string annotatedProgName;
  };
  
  /*! an actual instance of a raygen program - defined by its type and
      variable values */
  struct RayGen : public SBTObject<RayGenType> {
    typedef std::shared_ptr<RayGen> SP;

    /*! device-side representation of the actual variables for this
        program - unlike hit groups (which get written into the SBT)
        each raygen maintains its own SBT record */
    struct DeviceData : public RegisteredObject::DeviceData {
      DeviceData(const DeviceContext::SP &device, size_t dataSize);
      
      /*! device side copy of 'hostMemory' - this is the pointer that
          will go into the actual SBT */
      DeviceMemory         sbtRecordBuffer;

      /*! size of the SBT entry for this raygen program, with proper
          alignment and padding */
      const size_t         rayGenRecordSize;
    };

    /*! create new raygen of given type */
    RayGen(Context *const context,
           RayGenType::SP type);

    /*! clean up... */
    virtual ~RayGen();
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! execute a *synchronous* launch of this raygen program, of
      given dimensions - this will wait for the program to complete */
    void launch(const vec3i &dims);

    /*! *launch* this raygen prog with given launch params, but do NOT
         wait for completion - this means the SBT shuld NOT be changed
         or rebuild until a launchParams->sync() has been done */
    void launchAsync(const vec3i &dims, const LaunchParams::SP &launchParams);

    /*! *launch* this raygen prog with given launch params, but only for the 
         given device, and do NOT wait for completion. Launches can be done
         on a device by device basis for dynamic load balancing. */
    void launchAsyncOnDevice(const vec3i &dims,
                             uint32_t deviceID,
                             const LaunchParams::SP &launchParams);
    
    /*! write the given SBT record, using the given device's
        corresponding device-side data represenataion */
    void writeSBTRecord(uint8_t *const sbtRecord, const DeviceContext::SP &device);

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline RayGenType::DeviceData &
  RayGenType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<RayGenType::DeviceData>();
  }

  /*! get reference to given device-specific data for this object */
  inline RayGen::DeviceData &
  RayGen::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<RayGen::DeviceData>();
  }
} // ::owl

