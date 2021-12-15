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
#include "Geometry.h"
// #include "ll/DeviceMemory.h"
// #include "ll/Device.h"

namespace owl {

  /*! abstract base class for any sort of group (ie, BVH), BLAS'es and
      IAS'es will be derived from this class */
  struct Group : public RegisteredObject {
    typedef std::shared_ptr<Group> SP;

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc; for accel's that's pretty much the BVH memory
        for this type, and the specific OptixTraversableHandle to
        traverse it */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! constructor - pass-through to parent class */
      DeviceData(const DeviceContext::SP &device);

      /*! the handle for this BVH that can be passed to optixTrace */
      OptixTraversableHandle traversable = 0;

      /*! device memory that keeps the final, possibly compacted, BVH
          memory */
      DeviceMemory           bvhMemory;

      //! memory used for the BVH, last time it was built.
      size_t memFinal = 0;
      
      //! peak memory uesd during building, last time it was built.
      size_t memPeak = 0;
    };

    /*! constructor, that registers this group in the context's registry */
    Group(Context *const context,
          ObjectRegistry &registry);

    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! re*build* this accel - actual work depens on subclass */
    virtual void buildAccel() = 0;
    
    /*! re*fit* this accel - actual work depens on subclass */
    virtual void refitAccel() = 0;
    
    /*! return the SBT offset (ie, the offset at which the geometries
        within this group will be written into the Shader Binding
        Table) */
    virtual int getSBTOffset() const = 0;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! returns the (device-specific) optix traversable handle to traverse this group */
    inline OptixTraversableHandle getTraversable(const DeviceContext::SP &device) const;

    /*! returns the (device) memory used for this group's acceleration
      structure (but _excluding_ the memory for the geometries
      itself). "memFinal" is how much memory is used for the _final_
      version of the BVH (after it is done building), "memPeak" is peak
      memory used during construction. passing a NULL pointer to any
      value is valid; these values will get ignored. */
    void getAccelSize(size_t &memFinal, size_t &memPeak)
    {
      DeviceData &dd = deviceData[0]->as<DeviceData>();
      memFinal = dd.memFinal;
      memPeak  = dd.memPeak;
    }

    /*! bounding box for t=0 and t=1; for motion blur. */
    box3f bounds[2];
  };

  /*! a group containing geometries (ie, BLASes, whereas the
      InstanceGroup is a IAS */
  struct GeomGroup : public Group {
    typedef std::shared_ptr<GeomGroup> SP;

    /*! constructor for given number of chilren, will allocate the SBT
        range for those children*/
    GeomGroup(Context *const context, size_t numChildren);

    /*! destructor that releases the SBT range used by this group */
    virtual ~GeomGroup();

    /*! set given child ID to given geometry */
    void setChild(size_t childID, Geom::SP child);
    
    /*! return the SBT offset (ie, the offset at which the geometries
        within this group will be written into the Shader Binding
        Table) */
    int getSBTOffset() const override { return sbtOffset; }
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const;
    
    /*! list of child geometries to use in this BVH */
    std::vector<Geom::SP> geometries;

    /*! the SBT offset that this group will use to write its children
        into the SBT */
    const int sbtOffset;
  };

  
  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline Group::DeviceData &Group::getDD(const DeviceContext::SP &device) const
  {
    assert(device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }
  
  /*! returns the (device-specific) optix traversable handle to traverse this group */
  inline OptixTraversableHandle Group::getTraversable(const DeviceContext::SP &device) const
  { return getDD(device).traversable; }
  
} // ::owl
