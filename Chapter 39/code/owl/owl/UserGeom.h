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

#include "Geometry.h"

namespace owl {

  /*! Describes a geometry type for "User Geometries" or, as they area
    also known in OptiX land, "Custom Primitives" - ie, primitives
    where the user specifies bounding box program and intersection
    program. This "Type" class describes these programs; the
    "UserGeom" class then creates objects of this type and stores
    the respective data that describe that object */
  struct UserGeomType : public GeomType {
    typedef std::shared_ptr<UserGeomType> SP;

    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public GeomType::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! construct a new device-data for this type */
      DeviceData(const DeviceContext::SP &device);

      /*! fill in an OptixProgramGroup descriptor with the module and
        program names for this type */
      void fillPGDesc(OptixProgramGroupDesc &pgDesc,
                      GeomType *gt,
                      int rayType) override;

      /*! cuda function handle for the (automatically generatd) kernel
        that runs the primitive bounds program on the device */
      CUfunction boundsFuncKernel = 0;
    };

    /*! constructor, using the variable declaratoins that the user
      supplied */
    UserGeomType(Context *const context,
                 size_t varStructSize,
                 const std::vector<OWLVarDecl> &varDecls);

    /*! set intersection program to run for this type and given ray type */
    void setIntersectProg(int rayType,
                          Module::SP module,
                          const std::string &progName);
    
    /*! set bounding box program to run for this type */
    void setBoundsProg(Module::SP module,
                       const std::string &progName);

    /*! build the CUDA bounds program kernel (if bounds prog is set) */
    void buildBoundsProg();

    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! create an instance of this geometry that the user can then
      parameterize and attach to a group */
    virtual std::shared_ptr<Geom> createGeom() override;

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
    
    /*! create this object's device-specific data for the device */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! the bounds prog to run for this type */
    ProgramDesc boundsProg;
    
    /*! the vector of intersect programs to run for this type, one per
      ray type */
    std::vector<ProgramDesc> intersectProg;
  };


  /*! instance of a user geometry - this describes a type of geometry
    whose bounds and isec programs are speciffied through its
    associated type, and whose values are stores in this object's
    variables */
  struct UserGeom : public Geom {
    typedef std::shared_ptr<UserGeom> SP;

    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public Geom::DeviceData {
      DeviceData(const DeviceContext::SP &device)
        : Geom::DeviceData(device)
      {};

      /*! stors the device-side buffer to store the bounding boxes in
        that the bounds program generates, and that the BVH builder
        requires. (in theory we can release this memory after BVH is
        built)*/
      DeviceMemory internalBufferForBoundsProgram;
    };

    /*! constructor */
    UserGeom(Context *const context,
             GeomType::SP geometryType);

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device);
                        
    /*! get reference to the device-specific data for this object's *type* descriptor */
    UserGeomType::DeviceData &getTypeDD(const DeviceContext::SP &device) const;

    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! set number of primitives that this geom will contain */
    void setPrimCount(size_t count);
    
    /*! call a cuda kernel that computes the bounds *across* all
      primitives within this group; may only get caleld after bound
      progs have been executed */
    void computeBounds(box3f bounds[2]);

    /*! run the bounding box program for all primitives within this geometry */
    void executeBoundsProgOnPrimitives(const DeviceContext::SP &device);

    /*! number of prims that this geom will contain */
    size_t primCount = 0;
  };


  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline UserGeomType::DeviceData &
  UserGeomType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<UserGeomType::DeviceData>();
  }

  /*! get reference to given device-specific data for this object */
  inline UserGeom::DeviceData &
  UserGeom::getDD(const DeviceContext::SP &device)
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<UserGeom::DeviceData>();
  }

  /*! get reference to the device-specific data for this object's *type* descriptor */
  inline UserGeomType::DeviceData &
  UserGeom::getTypeDD(const DeviceContext::SP &device) const
  {
    return (UserGeomType::DeviceData &)type->getDD(device);
  }
} // ::owl
