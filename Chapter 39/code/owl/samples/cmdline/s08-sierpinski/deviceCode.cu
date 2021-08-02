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

using namespace owl;

#define NUM_SAMPLES_PER_PIXEL 16

// ----------- "triangle+material" -----------
template<typename LambertianTriangleMesh>
inline __device__
void closestHitPyramidMesh()
{
  const auto &self
    = owl::getProgramData<LambertianTriangleMesh>();
  PerRayData &prd = owl::getPRD<PerRayData>();

  // ID of the triangle we've hit:
  const int primID = optixGetPrimitiveIndex();
  
  const vec3i index = self.index[primID];
  const vec3f &A    = self.vertex[index.x];
  const vec3f &B    = self.vertex[index.y];
  const vec3f &C    = self.vertex[index.z];
  const vec3f N     = normalize(cross(B-A,C-A));
  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  
  const auto &material = *self.material;
  prd.out.scatterEvent
    = scatter(material,
              hit_P,N,
              prd)
    ? rayGotBounced
    : rayGotCancelled;
}

// ==================================================================
// actual closest-hit program instantiations for geom+material types
// ==================================================================

OPTIX_CLOSEST_HIT_PROGRAM(PyramidMesh)()
{ closestHitPyramidMesh<LambertianTriangleMesh>(); }

// ==================================================================
// miss and raygen
// ==================================================================

inline __device__
vec3f missColor(const Ray &ray)
{
  const vec2i pixelID = owl::getLaunchIndex();
  const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_MISS_PROGRAM(miss)()
{
  PerRayData &prd = owl::getPRD<PerRayData>();
  prd.out.scatterEvent = rayDidntHitAnything;
}

inline __device__
vec3f tracePath(const RayGenData &self,
                owl::Ray &ray, PerRayData &prd)
{
  vec3f attenuation = 1.f;
  
  /* iterative version of recursion, up to depth 50 */
  for (int depth=0;depth<5;depth++) {
    owl::traceRay(/*accel to trace against*/self.world,
                  /*the ray to trace*/ ray,
                  /*prd*/prd);
    if (prd.out.scatterEvent == rayDidntHitAnything) 
      /* ray got 'lost' to the environment - 'light' it with miss
         shader */
      return attenuation * missColor(ray);
    else if (prd.out.scatterEvent == rayGotCancelled) 
      return vec3f(0.f);

    else { // ray is still alive, and got properly bounced
      attenuation *= prd.out.attenuation;
      ray = owl::Ray(/* origin   : */ prd.out.scattered_origin,
                     /* direction: */ prd.out.scattered_direction,
                     /* tmin     : */ 1e-6f,
                     /* tmax     : */ 1e10f);
    }
  }
  // recursion did not terminate - cancel it
  return vec3f(0.f);
}

OPTIX_RAYGEN_PROGRAM(renderFrame)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  
  const int pixelIdx = pixelID.x+self.fbSize.x*(self.fbSize.y-1-pixelID.y);

  PerRayData prd;
  prd.random.init(pixelID.x,pixelID.y);

  vec3f color = 0.f;
  for (int sampleID=0;sampleID<NUM_SAMPLES_PER_PIXEL;sampleID++) {
    owl::Ray ray;

    const vec2f pixelSample(prd.random(),prd.random());
    const vec2f screen
      = (vec2f(pixelID)+pixelSample)
      / vec2f(self.fbSize);
    const vec3f origin = self.camera.origin; // + lens_offset;
    const vec3f direction
      = self.camera.lower_left_corner
      + screen.u * self.camera.horizontal
      + screen.v * self.camera.vertical
      - self.camera.origin;
  
    ray.origin = origin;
    ray.direction = normalize(direction);
  
    color += tracePath(self, ray, prd);
  }    
  
  self.fbPtr[pixelIdx]
    = owl::make_rgba(color * (1.f / NUM_SAMPLES_PER_PIXEL));
}
