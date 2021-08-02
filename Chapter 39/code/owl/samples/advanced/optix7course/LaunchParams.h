// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#include "Model.h"
#include "optix7.h"

namespace osc {
  using namespace owl;

  // for this simple example, we have a single ray type
  enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

  struct TriangleMeshSBTData {
    vec3f  color;
    vec3f *vertex;
    vec3f *normal;
    vec2f *texcoord;
    vec3i *index;
    int    hasTexture;
    cudaTextureObject_t texture;
  };
  
  struct LaunchParams
  {
    int numPixelSamples = 1;
    struct {
      int       frameID = 0;
      // the *final* frame buffer, after accum
      uint32_t    *fbFinal;
      // the color buffer, for accum buffering
      float4      *fbColor;
      // float4   *colorBuffer;
      // float4   *normalBuffer;
      // float4   *albedoBuffer;
      
      /*! the size of the frame buffer to render */
      vec2i     fbSize;
    } frame;
    
    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    struct {
      vec3f origin, du, dv, power;
    } light;
    
    OptixTraversableHandle traversable;
  };

} // ::osc
