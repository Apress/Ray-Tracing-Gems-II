/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NV_MISC_INCLUDED
#define NV_MISC_INCLUDED

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>

#include "nvprint.hpp"


/** 
  # functions in nvh

  - mipMapLevels : compute number of mip maps
  - stringFormat : sprintf for std::string
  - frand : random float using rand()
  - permutation : fills uint vector with random permutation of values [0... vec.size-1]

*/

namespace nvh
{

  inline std::string stringFormat(const char* msg, ...)
  {
    char text[8192];
    va_list list;

    if (msg == 0)
      return std::string();

    va_start(list, msg);
    vsnprintf(text, sizeof(text), msg, list);
    va_end(list);

    return std::string(text);
  }

  inline float frand(){
    return float( rand() % RAND_MAX ) / float(RAND_MAX);
  }

  inline int mipMapLevels(int size) {
    int num = 0;
    while (size){
      num++;
      size /= 2;
    }
    return num;
  }

  // permutation creates a random permutation of all integer values 
  // 0..data.size-1 occuring once within data.

  inline void permutation(std::vector<unsigned int> &data)
  {
    size_t size = data.size();
    assert( size < RAND_MAX );

    for (size_t i = 0; i < size; i++){
      data[i] = (unsigned int)(i);
    }

    for (size_t i = size-1; i > 0 ; i--){
      size_t other = rand() % (i+1);
      std::swap(data[i],data[other]);
    }
  }
}

#endif
