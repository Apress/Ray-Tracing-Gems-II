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

#include "RegisteredObject.h"

namespace owl {
  
  /*! captures the concept of a module that contains one or more
    programs. */
  struct Module : public RegisteredObject {
    typedef std::shared_ptr<Module> SP;

    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public RegisteredObject::DeviceData {
      
      /*! constructor */
      DeviceData(Module *parent, DeviceContext::SP device);
      
      /*! destructor */
      virtual ~DeviceData();

      /*! build the optix side of this module on this device */
      void build();

      /*! destroy the optix data for this module; the owl data for the
        module itself remains valid */
      void destroy();

      /*! pointer to the non-device specific part of this module */
      Module *const parent;
      
      /*! optix-compiled module for the optix programs. for all
       *optix* programs we can directly build the PTX code into a
       module using optixbuildmodule - this is the result of that
       operation */
      OptixModule module = 0;
      
      /*! for the *bounds* function we have to build a *separate*
        module because this one is built outside of optix, and thus
        does not have the internal _optix_xyz() symbols in it */
      CUmodule    boundsModule = 0;
    };

    /*! constructor - ptxCode contains the prec-ompiled ptx code with
      the compiled functions */
    Module(Context *context, const std::string &ptxCode);

    /*! destructor, to release data if required */
    virtual ~Module();
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! create this object's device-specific data for the device */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! the precompiled PTX code supplied by the user */
    const std::string ptxCode;
  };
  
  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline Module::DeviceData &Module::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<Module::DeviceData>();
  }
  
} // ::owl
