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

#ifndef NV_GLSLTYPES_INCLUDED
#define NV_GLSLTYPES_INCLUDED

/**
  # type definitions for nvgl and nvmath
  Sets up vector, matrix etc. types available in GLSL
*/

#include <stdint.h>
#include <nvmath/nvmath_glsltypes.h>

namespace nvgl
{
  typedef uint64_t sampler1D;
  typedef uint64_t sampler2D;
  typedef uint64_t sampler2DMS;
  typedef uint64_t sampler3D;
  typedef uint64_t samplerBuffer;
  typedef uint64_t samplerCube;
  typedef uint64_t sampler1DArray;
  typedef uint64_t sampler2DArray;
  typedef uint64_t sampler2DMSArray;
  typedef uint64_t samplerCubeArray;

  typedef uint64_t usampler1D;
  typedef uint64_t usampler2D;
  typedef uint64_t usampler2DMS;
  typedef uint64_t usampler3D;
  typedef uint64_t usamplerBuffer;
  typedef uint64_t usamplerCube;
  typedef uint64_t usampler1DArray;
  typedef uint64_t usampler2DArray;
  typedef uint64_t usampler2DMSArray;
  typedef uint64_t usamplerCubeArray;
  
  typedef uint64_t isampler1D;
  typedef uint64_t isampler2D;
  typedef uint64_t isampler2DMS;
  typedef uint64_t isampler3D;
  typedef uint64_t isamplerBuffer;
  typedef uint64_t isamplerCube;
  typedef uint64_t isampler1DArray;
  typedef uint64_t isampler2DArray;
  typedef uint64_t isampler2DMSArray;
  typedef uint64_t isamplerCubeArray;
  
  typedef uint64_t image1D;
  typedef uint64_t image2D;
  typedef uint64_t image2DMS;
  typedef uint64_t image3D;
  typedef uint64_t imageBuffer;
  typedef uint64_t imageCube;
  typedef uint64_t image1DArray;
  typedef uint64_t image2DArray;
  typedef uint64_t image2DMSArray;
  typedef uint64_t imageCubeArray;

  typedef uint64_t uimage1D;
  typedef uint64_t uimage2D;
  typedef uint64_t uimage2DMS;
  typedef uint64_t uimage3D;
  typedef uint64_t uimageBuffer;
  typedef uint64_t uimageCube;
  typedef uint64_t uimage1DArray;
  typedef uint64_t uimage2DArray;
  typedef uint64_t uimage2DMSArray;
  typedef uint64_t uimageCubeArray;
  
  typedef uint64_t iimage1D;
  typedef uint64_t iimage2D;
  typedef uint64_t iimage2DMS;
  typedef uint64_t iimage3D;
  typedef uint64_t iimageBuffer;
  typedef uint64_t iimageCube;
  typedef uint64_t iimage1DArray;
  typedef uint64_t iimage2DArray;
  typedef uint64_t iimage2DMSArray;
  typedef uint64_t iimageCubeArray;
}

#endif
