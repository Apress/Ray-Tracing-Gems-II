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

#include "../s05-rtow/Materials.h"

using namespace owl;

struct LambertianTriangleMesh {
  /*! for now, let's use a single material for all pyramid triangles
      ... .*/
  Lambertian *material;
  /* the vertex and index arrays for the triangle mesh */
  vec3f *vertex;
  vec3i *index;
};

struct RayGenData
{
  OptixTraversableHandle world;
  uint32_t *fbPtr;
  vec2i     fbSize;
  
  struct {
    vec3f origin;
    vec3f lower_left_corner;
    vec3f horizontal;
    vec3f vertical;
  } camera; 
};

struct MissProgData
{
  /* nothing */
};

