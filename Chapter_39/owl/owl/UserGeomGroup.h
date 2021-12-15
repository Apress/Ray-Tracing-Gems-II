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
#include "UserGeom.h"

namespace owl {

  /*! a geometry group / BLAS over user-defined/custom
      primitives. This Group may only be built over UserGeom's */
  struct UserGeomGroup : public GeomGroup {

    UserGeomGroup(Context *const context,
                   size_t numChildren,
                   unsigned int buildFlags);
    virtual std::string toString() const { return "UserGeomGroup"; }

    /*! build() and refit() share most of their code; this functoin
        does all that code, with only minor specialization based on
        build vs refit */
    void buildOrRefit(bool FULL_REBUILD);
    
    void buildAccel() override;
    void refitAccel() override;

    /*! low-level accel structure builder for given device */
    template<bool FULL_REBUILD>
    void buildAccelOn(const DeviceContext::SP &device);

    constexpr static unsigned int defaultBuildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

    protected:
    const unsigned int buildFlags;

  };

} // ::owl
