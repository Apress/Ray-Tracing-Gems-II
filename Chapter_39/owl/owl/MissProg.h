// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

  /*! type that describes a miss program's variables and programs */
  struct MissProgType : public SBTObjectType {
    typedef std::shared_ptr<MissProgType> SP;
    MissProgType(Context *const context,
               Module::SP module,
               const std::string &progName,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    /*! for miss progs there's exactly one programgroup pre object */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;
      
      /*! constructor, only pass-through to parent class */
      DeviceData(const DeviceContext::SP &device);

      /*! the optix-compiled program group witin the given device's
        optix context */
      OptixProgramGroup pg = 0;
    };

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
    
    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
    
    /*! the module in which the program is defined */
    Module::SP module;

    /*! name of the program within this module */
    const std::string progName;
    /*! the name, annotated wiht optix' "__miss__" */
    const std::string annotatedProgName;
  };
  
  /*! an actual instance of a miss program - defined by its type and
      variable values */
  struct MissProg : public SBTObject<MissProgType> {
    typedef std::shared_ptr<MissProg> SP;

    /*! constructor */
    MissProg(Context *const context,
           MissProgType::SP type);
    
    /*! write the given SBT record, using the given device's
      corresponding device-side data represenataion */
    void writeSBTRecord(uint8_t *const sbtRecord,
                        const DeviceContext::SP &device);
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
  };
  
  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline MissProgType::DeviceData &MissProgType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }
  
} // ::owl

