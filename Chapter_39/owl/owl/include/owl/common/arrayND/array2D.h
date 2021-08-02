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

#include <owl/common/math/vec.h>

namespace owl {
  namespace common {
    namespace array2D {
    
      inline int linear(const vec2i &ID, const vec2i &dims)
      { return ID.x + dims.x*ID.y; }

      template<typename Lambda>
      inline void for_each(const vec2i &dims, const Lambda &lambda)
      {
        for (int iy=0;iy<dims.y;iy++)
          for (int ix=0;ix<dims.x;ix++)
            lambda(vec2i(ix,iy));
      }

      template<typename Lambda>
      inline void for_each(const vec2i &begin,
                           const vec2i &end,
                           const Lambda &lambda)
      {
        for (int iy=begin.y;iy<end.y;iy++)
          for (int ix=begin.x;ix<end.x;ix++)
            lambda(vec2i(ix,iy));
      }

// #if OWL_HAVE_PARALLEL_FOR
      template<typename Lambda>
      inline void parallel_for(const vec2i &dims, const Lambda &lambda)
      {
        owl::common::parallel_for(dims.x*dims.y,[&](int index){
            lambda(vec2i(index%dims.x,index/dims.x));
          });
      }
// #endif
      template<typename Lambda>
      inline void serial_for(const vec2i &dims, const Lambda &lambda)
      {
        owl::common::serial_for(dims.x*dims.y,[&](int index){
            lambda(vec2i(index%dims.x,index/dims.x));
          });
      }
    
      template<typename Lambda>
      inline void parallel_for_blocked(const vec2i &dims,
                                       const vec2i &blockSize,
                                       const Lambda &lambda)
      {
        const vec2i numBlocks = divRoundUp(dims,blockSize);
        array2D::parallel_for
          (numBlocks,[&](const vec2i &block){
                       const vec2i begin = block*blockSize;
                       const vec2i end   = min(begin+blockSize,dims);
                       lambda(begin,end);
                       // array2D::for_each(begin,end,
                       //                   [&](const vec2i &pixel)
                       //                   { lambda(pixel); });
                     });
      }
    } // owl::common::array2D
  } // owl::common
} // owl
