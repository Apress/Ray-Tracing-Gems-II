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
#include "SBTObject.h"
#include "Module.h"
#include "Buffer.h"

namespace owl {

  struct Geom;

  /*! describes the two components of any device program: the module
      with the precompiled PTX code that contains the program, and the
      name of the program within this module */
  struct ProgramDesc {
    Module::SP  module;
    std::string progName;
  };

  /* abstraction for any sort of geometry type - describes the
     programs to use, and structure of the SBT records, when building
     shader binding tables (SBTs) with geometries of this type. This
     will later get subclassed into triangle geometries, user/custom
     primtivie geometries, etc */
  struct GeomType : public SBTObjectType {
    typedef std::shared_ptr<GeomType> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc - for geomtypes that's pretty much the
        OptixProgramGroup's that describe the precompiled programs to
        use for geometries of this type*/
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! constructor - passthrough to parent class */
      DeviceData(const DeviceContext::SP &device);

      /*! build the optix program groups for the given (parent)
          geomtype */
      void buildHitGroupPrograms(GeomType *gt);
      
      /*! fill in an OptixProgramGroup descriptor with the module and
          program names for this type */
      virtual void fillPGDesc(OptixProgramGroupDesc &pgDesc,
                              GeomType *gt, int rayType);
      
      /*! hit group program groups, per ray type */
      std::vector<OptixProgramGroup> hgPGs;
    };

    /*! create new geometry type with given parameters/variables */
    GeomType(Context *const context,
             /*! size of the struct on the device */
             size_t varStructSize,
             /*! list of variables within this struct that we can/have to set */
             const std::vector<OWLVarDecl> &varDecls);
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    
    /*! create an instance of tihs geometry - abstract, since the
        actual class to create depends on what *kind* of geometry this
        is */
    virtual std::shared_ptr<Geom> createGeom() = 0;

    /*! sets the anyhit program to run for given ray type */
    void setAnyHitProgram(int rayType,
                          Module::SP module,
                          const std::string &progName);
    
    /*! sets the closest program to run for given ray type */
    void setClosestHitProgram(int rayType,
                              Module::SP module,
                              const std::string &progName);

    /*! closest programs to run for this geom - one per ray type */
    std::vector<ProgramDesc> closestHit;
    
    /*! anyhit programs to run for this geom - one per ray type */
    std::vector<ProgramDesc> anyHit;
  };

  /*! a actual geometry object with primitives - this class is still
      abstract, and will get fleshed out in its derived classes
      (UserGeom, TrianlgesGeom, ...) */
  struct Geom : public SBTObject<GeomType> {
    typedef std::shared_ptr<Geom> SP;

    /*! constructor - mostly pass through to parent class */
    Geom(Context *const context, GeomType::SP geomType);
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! write the SBT record for this object and ray type; this
        requires finding the proper programs (from the type and ray
        type), and writign the variables */
    void writeSBTRecord(/*! where to write to */
                        uint8_t *const sbtRecord,
                        /*! device for which we need to write the
                            device represenatation */
                        const DeviceContext::SP &device,
                        /*! the ray type that defines which programs
                            to use */
                        int rayTypeID);

    /*! the geometry type that desribes this geometry's variables and
        programs */
    GeomType::SP geomType;
  };
  
  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline GeomType::DeviceData &GeomType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }
  
} // ::owl
