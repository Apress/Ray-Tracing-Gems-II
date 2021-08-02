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

#include "DeviceContext.h"

namespace owl {

  /*! convert a OWLDataType enum into a strict that represents the name of that type */
  std::string typeToString(OWLDataType type);

  /*! returns number of bytes for given data type (where applicable) */
  size_t      sizeOf(OWLDataType type);


  /*! common "root" abstraction for every object this library creates */
  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
    struct DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! construtor */
      DeviceData(DeviceContext::SP device) : device(device) {};

      /*! destructor - does nothing in itself, but forces a virtual
          destructor for derived classes */
      virtual ~DeviceData() {}

      /*! pretty-typecast into derived classes */
      template<typename T> inline T &as();

      /*! shared-pointer to the device context in wihch this
          device-specific data lives; makes sure that all 'dependent'
          device data can properly destruct before the device context
          itself dies */
      DeviceContext::SP device;
    };

    /*! constructor */
    Object();

    /*! pretty-printer, for printf-debugging */
    virtual std::string toString() const;

    /*! creates the device-specific data for this group */
    virtual DeviceData::SP createOn(const std::shared_ptr<DeviceContext> &device);

    /*! creates the actual device data for all devies,by calling \see
        createOn() for each device */
    void createDeviceData(const std::vector<std::shared_ptr<DeviceContext>> &devices);

    /*! pretty-typecase to all derived classes */
    template<typename T> inline std::shared_ptr<T> as();

    /*! a unique ID we assign to each newly created object - this
        allows any caching algorithms to check if a given object was
        replaced with another, even if in some internal array it ends
        up with the same array index as a previous other object */
    const size_t uniqueID;

    /*! atomic counter that always describes the next not yet used
        unique ID, which we can use to fill in the Object::uniquID
        values */
    static std::atomic<uint64_t> nextAvailableID;

    /*! the list of per-device data for this object - should be
        exactly one per GPU in the context */
    std::vector<DeviceData::SP> deviceData;
  };

  
  /*! a object that belongs to a context */
  struct ContextObject : public Object {
    typedef std::shared_ptr<ContextObject> SP;
    
    ContextObject(Context *const context)
      : context(context)
    {}
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
    
    Context *const context;
  };


  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! pretty-typecast into derived classes */
  template<typename T> inline T &Object::DeviceData::as()
  { return *dynamic_cast<T *>(this); }

  /*! pretty-typecase to all derived classes */
  template<typename T> inline std::shared_ptr<T> Object::as() 
  { return std::dynamic_pointer_cast<T>(shared_from_this()); }

} // ::owl

