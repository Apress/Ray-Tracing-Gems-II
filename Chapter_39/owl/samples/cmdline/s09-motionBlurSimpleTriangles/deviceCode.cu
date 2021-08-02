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

#include "deviceCode.h"
#include <owl/common/math/random.h>
#include <optix_device.h>

typedef owl::common::LCG<4> Random;

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program%s\n",
           OWL_TERMINAL_CYAN,
           OWL_TERMINAL_DEFAULT);
  }

  Random rng(pixelID);
  
  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);
  
  vec3f avgColor = 0.f;
  const int numSPP = 16;
  for (int i=0;i<numSPP;i++) {
    ray.time = rng();
    vec3f color;
    owl::traceRay(/*accel to trace against*/self.world,
                  /*the ray to trace*/ray,
                  /*prd*/color);
    
    avgColor += color;
  }
  avgColor *= 1.f/numSPP;
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
    self.fbPtr[fbOfs]
      = owl::make_rgba(avgColor);
  }
  
inline __device__ vec3f lerp(const vec3f &A, const vec3f &B, float t)
{ return (1.f-t)*A + t*B; }
  
OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  // compute normal:
  const int   primID  = optixGetPrimitiveIndex();
  const float time    = optixGetRayTime();
  const vec3i index   = self.index[primID];
  const vec3f &A0     = self.vertex0[index.x];
  const vec3f &B0     = self.vertex0[index.y];
  const vec3f &C0     = self.vertex0[index.z];
  const vec3f &A1     = self.vertex1[index.x];
  const vec3f &B1     = self.vertex1[index.y];
  const vec3f &C1     = self.vertex1[index.z];
  const vec3f A       = lerp(A0,A1,time);
  const vec3f B       = lerp(B0,B1,time);
  const vec3f C       = lerp(C0,C1,time);
  const vec3f Ng      = normalize(cross(B-A,C-A));

  const vec3f rayDir  = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}

