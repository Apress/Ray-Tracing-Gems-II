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
#include <optix_device.h>

__constant__ Globals optixLaunchParams;


struct Hit {
  bool  hadHit = false;
  vec3f pos, nor, col;
};

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  int targetDeviceIndex
    = (getLaunchIndex().x / 32) % optixLaunchParams.deviceCount;
  if (targetDeviceIndex != optixLaunchParams.deviceIndex)
    return;
  
  const vec2i pixelID = owl::getLaunchIndex();

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

  Hit hit;
  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/hit);

  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(hit.col);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  Hit &prd = owl::getPRD<Hit>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f rayDir = normalize((vec3f)optixGetWorldRayDirection());

  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec4i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  vec3f Ng     = normalize(cross(B-A,C-A));
  Ng = optixTransformNormalFromObjectToWorldSpace(Ng);
  if (dot(Ng,rayDir) > 0.f) Ng = -Ng;

  const vec2f uv     = optixGetTriangleBarycentrics();
  const vec2f tc
    = (1.f-uv.x-uv.y)*self.texCoord[index.x]
    +      uv.x      *self.texCoord[index.y]
    +           uv.y *self.texCoord[index.z];

  device::Buffer *textureSets =
    (device::Buffer *)self.textureSets.data;
  uint32_t textureSetID
    = uint32_t(optixLaunchParams.time)
    % NUM_TEXTURE_SETS;
  cudaTextureObject_t *textureSet
    = (cudaTextureObject_t *)textureSets[textureSetID].data;
  cudaTextureObject_t texture
    = textureSet[index.w];
  vec4f texColor = tex2D<float4>(texture,tc.x,tc.y);

  const vec3f P = rayOrg + (optixGetRayTmax()*.999f) * rayDir;

  const vec3f lightDir(1,1,1);
  bool illuminated = false;
  if (dot(lightDir,Ng) > 0.f) {
    illuminated = true;
  }

  float weight = .1f;
  weight += .2f*fabs(dot(rayDir,Ng));
  if (illuminated)
    weight += 1.5f * dot(normalize(lightDir),Ng);
  
  prd.col = weight * vec3f(texColor);
  prd.hadHit = true;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  Hit &prd = owl::getPRD<Hit>();
  const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  prd.hadHit = false;
  prd.col = c;
}

