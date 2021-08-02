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

#include "owl/common/math/vec.h"
#include "owl/common/parallel/parallel_for.h"

namespace owl {
  namespace common {
    namespace array3D {
    
      inline __both__ int64_t linear(const vec3i &ID,
                                     const vec3i &dims)
      { return ID.x + dims.x*(ID.y + dims.y*(int64_t)ID.z); }

      template<typename Lambda>
      inline  void for_each(const vec3i &dims,
                            const Lambda &lambda)
      {
        for (int iz=0;iz<dims.z;iz++)
          for (int iy=0;iy<dims.y;iy++)
            for (int ix=0;ix<dims.x;ix++)
              lambda(vec3i(ix,iy,iz));
      }

      template<typename Lambda>
      inline  void for_each(const vec3i &begin,
                            const vec3i &end,
                            const Lambda &lambda)
      {
        for (int iz=begin.z;iz<end.z;iz++)
          for (int iy=begin.y;iy<end.y;iy++)
            for (int ix=begin.x;ix<end.x;ix++)
              lambda(vec3i(ix,iy,iz));
      }

      template<typename Lambda>
      inline void parallel_for(const vec3i &dims, const Lambda &lambda)
      {
        owl::common::parallel_for
          (dims.x*(size_t)dims.y*dims.z,[&](size_t index){
                                          lambda(vec3i(index%dims.x,
                                                       (index/dims.x)%dims.y,
                                                       index/((size_t)dims.x*dims.y)));
                                        });
      }
      template<typename Lambda>
      inline  void serial_for(const vec3i &dims, const Lambda &lambda)
      {
        owl::common::serial_for(dims.x*size_t(dims.y)*dims.z,
                                [&](size_t index){
                                  lambda(vec3i(index%dims.x,
                                               (index/dims.x)%dims.y,
                                               index/((size_t)dims.x*dims.y)));
                                });
      }

      inline __both__ bool validIndex(const vec3i &idx,
                                      const vec3i &dims)
      {
        if (idx.x < 0 || idx.x >= dims.x) return false;
        if (idx.y < 0 || idx.y >= dims.y) return false;
        if (idx.z < 0 || idx.z >= dims.z) return false;
        return true;
      }
      
    } // owl::common::array3D
  } // owl::common
} // owl
