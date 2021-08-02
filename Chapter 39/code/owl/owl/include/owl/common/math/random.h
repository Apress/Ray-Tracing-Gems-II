// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

/* pieces originally taken from optixPathTracer/random.h example,
   under following license */

/* 
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "owl/common/math/vec.h"

namespace owl {
  namespace common {

    /*! simple 24-bit linear congruence generator */
    template<unsigned int N=4>
    struct LCG {
    
      inline __both__ LCG()
      { /* intentionally empty so we can use it in device vars that
           don't allow dynamic initialization (ie, PRD) */
      }
      inline __both__ LCG(unsigned int val0, unsigned int val1)
      { init(val0,val1); }

      inline __both__ LCG(const vec2i &seed)
      { init((unsigned)seed.x,(unsigned)seed.y); }
      inline __both__ LCG(const vec2ui &seed)
      { init(seed.x,seed.y); }
      
      inline __both__ void init(unsigned int val0, unsigned int val1)
      {
        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;
      
        for (unsigned int n = 0; n < N; n++) {
          s0 += 0x9e3779b9;
          v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
          v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
        }
        state = v0;
      }
    
      // Generate random unsigned int in [0, 2^24)
      inline __both__ float operator() ()
      {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return ldexpf(float(state), -32);
        // return (state & 0x00FFFFFF) / (float) 0x01000000;
      }
    
      uint32_t state;
    };


    /*! literal re-implementation of the stdlib 'drand48()' LCG
      generator. note this is usually significantly worse than the
      owl::common::LCG class above */
    struct DRand48
    {
      /*! initialize the random number generator with a new seed (usually
        per pixel) */
      inline __both__ void init(int seed = 0)
      {
        state = seed;
        for (int warmUp=0;warmUp<10;warmUp++)
          (*this)();
      }

      /*! get the next 'random' number in the sequence */
      inline __both__ float operator() ()
      {
        const uint64_t a = 0x5DEECE66DULL;
        const uint64_t c = 0xBULL;
        const uint64_t mask = 0xFFFFFFFFFFFFULL;
        state = a*state + c;
        return float(state & mask) / float(mask+1ULL);
        //return ldexpf(float(state & mask), -24);
      }

      uint64_t state;
    };
    
  } // ::owl::common
} // ::owl
