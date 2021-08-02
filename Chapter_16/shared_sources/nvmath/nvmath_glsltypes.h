/* Copyright (c) 2012-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NV_SHADER_TYPES_H
#define NV_SHADER_TYPES_H

#include "nvmath_types.h"
#include "NvFoundation.h"

namespace nvmath {

#if defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(__AMD64__)
// Matrices, must align to 4 vector (16 bytes)
NV_ALIGN(16, typedef mat3f mat3);
NV_ALIGN(16, typedef mat4f mat4);

// vectors, 4-tuples and 3-tuples must align to 16 bytes
//  2-vectors must align to 8 bytes
NV_ALIGN(16, typedef vec4f vec4);
NV_ALIGN(16, typedef vec3f vec3);
NV_ALIGN(8, typedef vec2f vec2);

NV_ALIGN(16, typedef vec4i ivec4);
NV_ALIGN(16, typedef vec3i ivec3);
NV_ALIGN(8, typedef vec2i ivec2);

NV_ALIGN(16, typedef vec4ui uvec4);
NV_ALIGN(16, typedef vec3ui uvec3);
NV_ALIGN(8, typedef vec2ui uvec2);
#else
// Matrices, must align to 4 vector (16 bytes)
typedef mat4f mat4;

// vectors, 4-tuples and 3-tuples must align to 16 bytes
//  2-vectors must align to 8 bytes
typedef vec4f vec4;
typedef vec3f vec3;
typedef vec2f vec2;

typedef vec4i ivec4;
typedef vec3i ivec3;
typedef vec2i ivec2;

typedef vec4ui uvec4;
typedef vec3ui uvec3;
typedef vec2ui uvec2;
#endif

//class to make uint look like bool to make GLSL packing rules happy
struct boolClass
{
  unsigned int _rep;

  boolClass()
      : _rep(false)
  {
  }
  boolClass(bool b)
      : _rep(b)
  {
  }
             operator bool() { return _rep == 0 ? false : true; }
  boolClass& operator=(bool b)
  {
    _rep = b;
    return *this;
  }
};

NV_ALIGN(4, typedef boolClass bool32);

}  // namespace nvmath

#endif
