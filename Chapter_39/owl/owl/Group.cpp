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

#include "Group.h"
#include "Context.h"

namespace owl {
  
  // ------------------------------------------------------------------
  // Group::DeviceData
  // ------------------------------------------------------------------
  
  /*! constructor - pass-through to parent class */
  Group::DeviceData::DeviceData(const DeviceContext::SP &device)
    : RegisteredObject::DeviceData(device)
  {}

  // ------------------------------------------------------------------
  // Group
  // ------------------------------------------------------------------
  
  /*! constructor, that registers this group in the context's registry */
  Group::Group(Context *const context,
               ObjectRegistry &registry)
    : RegisteredObject(context,registry)
  {}

  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP Group::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }
  
  /*! pretty-printer, for printf-debugging */
  std::string Group::toString() const
  {
    return "Group";
  }

  // ------------------------------------------------------------------
  // GeomGroup
  // ------------------------------------------------------------------

  /*! constructor for given number of chilren, will allocate the SBT
    range for those children*/
  GeomGroup::GeomGroup(Context *const context,
                       size_t numChildren)
    : Group(context,context->groups),
      geometries(numChildren),
      sbtOffset(context->sbtRangeAllocator.alloc(numChildren))
  {}
  
  /*! destructor that releases the SBT range used by this group */
  GeomGroup::~GeomGroup()
  {
    context->sbtRangeAllocator.release(sbtOffset,geometries.size());
  }


  /*! set given child ID to given geometry */
  void GeomGroup::setChild(size_t childID, Geom::SP child)
  {
    assert(childID < geometries.size());
    geometries[childID] = child;
  }
  
  /*! pretty-printer, for printf-debugging */
  std::string GeomGroup::toString() const
  {
    return "GeomGroup";
  }
  
} // ::owl
