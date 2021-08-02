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

#include "Group.h"

namespace owl {

  /*! a group / BLAS over triangle meshes */
  struct TrianglesGeomGroup : public GeomGroup {

    /*! constructor - passthroughto parent class */
    TrianglesGeomGroup(Context *const context, size_t numChildren);
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    void buildAccel() override;
    void refitAccel() override;

    /*! (re-)compute the Group::bounds[2] information for motion blur
      - ie, our _parent_ node may need this */
    void updateMotionBounds();

    /*! low-level accel structure builder for given device */
    template<bool FULL_REBUILD>
    void buildAccelOn(const DeviceContext::SP &device);
  };

} // ::owl
