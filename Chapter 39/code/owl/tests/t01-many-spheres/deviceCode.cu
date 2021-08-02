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

#include "GeomTypes.h"
#include <optix_device.h>

using namespace owl;

#define NUM_SAMPLES_PER_PIXEL 16

// ==================================================================
// bounding box programs - since these don't actually use the material
// they're all the same irrespective of geometry type, so use a
// template ...
// ==================================================================
template<typename SphereGeomType>
inline __device__ void boundsProg(const void *geomData,
                                  box3f &primBounds,
                                  const int primID)
{
  const SphereGeomType &self = *(const SphereGeomType*)geomData;
  const Sphere sphere = self.prims[primID].sphere;
  primBounds = box3f()
    .extend(sphere.center - sphere.radius)
    .extend(sphere.center + sphere.radius);
}

OPTIX_BOUNDS_PROGRAM(LambertianSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<LambertianSpheresGeom>(geomData,primBounds,primID); }

// ==================================================================
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================

template<typename SpheresGeomType>
inline __device__ void intersectProg()
{
  const int primID = optixGetPrimitiveIndex();
  const auto &self
    = owl::getProgramData<SpheresGeomType>().prims[primID];

  /* iw, jan 11, 2020: for this particular example (where we do not
     yet use instancing) we could also use the World ray; but this
     verion is cleaner since it would/will also work with
     instancing */
  const vec3f org  = optixGetObjectRayOrigin();
  const vec3f dir  = optixGetObjectRayDirection();
  const float tmin = optixGetRayTmin();
#if 0
  float hit_t      = optixGetRayTmax();

  const vec3f oc = org - self.sphere.center;
  const float a = dot(dir,dir);
  const float b = dot(oc, dir);
  const float c = dot(oc, oc) - self.sphere.radius * self.sphere.radius;
  const float discriminant = b * b - a * c;
  
  if (discriminant < 0.f) return;

  {
    float temp = (-b - sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin) 
      hit_t = temp;
  }
      
  {
    float temp = (-b + sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin) 
      hit_t = temp;
  }
  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
  }
#else

  // Raytracing Gem
  const float particleRadius = self.sphere.radius;
  const vec3f oc = org - self.sphere.center;
  const float sqrRad = particleRadius * particleRadius;
  
  //const float  a = dot(ray.direction, ray.direction);
  const float  b = dot(-oc, dir);
  const vec3f temp = oc + b * dir;
  const float  delta = sqrRad - dot(temp, temp);
  
  if (delta < 0.0f) return;

  const float c = dot(oc, oc) - sqrRad;
  const float sign = signbit(b) ? 1.0f : -1.0f;
  const float q = b + sign * sqrtf(delta);
  
  float hit_t = min(c / q, q);
  // printf("cspehre %i :  %f %f %f : %f hit_t %f\n",
  //        primID,
  //        self.sphere.center.x,
  //        self.sphere.center.y,
  //        self.sphere.center.z,
  //        particleRadius,
  //        hit_t);
  if (hit_t >= tmin && hit_t < optixGetRayTmax()) 
    optixReportIntersection(hit_t, 0);
#endif
}


OPTIX_INTERSECT_PROGRAM(LambertianSpheres)()
{ intersectProg<LambertianSpheresGeom>(); }

// ==================================================================
// plumbing for closest hit
// ==================================================================

/*! transform a _point_ from object to world space */
inline __device__ vec3f pointToWorld(const vec3f &P)
{
  return (vec3f)optixTransformPointFromObjectToWorldSpace(P);
}


template<typename SpheresGeomType>
inline __device__
void closestHit()
{
  const int primID = optixGetPrimitiveIndex();
  const auto &self
    = owl::getProgramData<SpheresGeomType>().prims[primID];

  PerRayData &prd = owl::getPRD<PerRayData>();

  const vec3f org  = optixGetWorldRayOrigin();
  const vec3f dir  = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  /* iw, jan 11, 2020: for this particular example (where we do not
     yet use instancing) we could also get away with just using the
     sphere.center value directly (since we don't use instancing the
     transform will not have any effect, anyway); but this verion is
     cleaner since it would/will also work with instancing */
  const vec3f N     = hit_P-pointToWorld(self.sphere.center);

  prd.out.scatterEvent
    = scatter(self.material,
              hit_P,N,//ray,
              prd)
    ? rayGotBounced
    : rayGotCancelled;
}

OPTIX_CLOSEST_HIT_PROGRAM(LambertianSpheres)()
{ closestHit<LambertianSpheresGeom>(); }




// ==================================================================
// miss and raygen
// ==================================================================

inline __device__
vec3f missColor(const Ray &ray)
{
  const vec2i pixelID = owl::getLaunchIndex();

  const vec3f rayDir = normalize(ray.direction);
  const float t = 0.5f*(rayDir.y + 1.0f);
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_MISS_PROGRAM(miss)()
{
  PerRayData &prd = owl::getPRD<PerRayData>();
  prd.out.scatterEvent = rayDidntHitAnything;
  // const vec2i pixelID = owl::getLaunchIndex();

  // const MissProgData &self = owl::getProgramData<MissProgData>();

  // const vec3f unit_direction = normalize((vec3f)optixGetWorldRayDirection());
  // const float t = 0.5f*(unit_direction.y + 1.0f);
  // const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  // vec3f &prd = owl::getPRD<vec3f>();
  // prd = c;
}



inline __device__
vec3f tracePath(const RayGenData &self,
                owl::Ray &ray, PerRayData &prd)
{
  vec3f attenuation = 1.f;
  
  /* iterative version of recursion, up to depth 50 */
  for (int depth=0;depth<50;depth++) {
    owl::trace(/*accel to trace against*/self.world,
               /*the ray to trace*/ ray,
               /*numRayTypes*/1,
               /*prd*/prd,
               self.sbtOffset);
    
    if (prd.out.scatterEvent == rayDidntHitAnything)
      /* ray got 'lost' to the environment - 'light' it with miss
         shader */
      return attenuation * missColor(ray);
    else if (prd.out.scatterEvent == rayGotCancelled)
      return vec3f(0.f);

    else { // ray is still alive, and got properly bounced
      attenuation *= prd.out.attenuation;
      ray = owl::Ray(/* origin   : */ prd.out.scattered_origin,
                     /* direction: */ normalize(prd.out.scattered_direction),
                     /* tmin     : */ 1e-3f,
                     /* tmax     : */ 1e10f);
    }
  }
  // recursion did not terminate - cancel it
  return vec3f(0.f);
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  
  const int pixelIdx = pixelID.x+self.fbSize.x*(self.fbSize.y-1-pixelID.y);

  // for multi-gpu: only render every deviceCount'th column of 32 pixels:
  if (((pixelID.x/32) % self.deviceCount) != self.deviceIndex)
    return;

  PerRayData prd;
  prd.random.init(pixelID.x,pixelID.y);
  
  vec3f color = 0.f;
  for (int sampleID=0;sampleID<NUM_SAMPLES_PER_PIXEL;sampleID++) {
    owl::Ray ray;
    
    const vec2f pixelSample(prd.random(),prd.random());
    const vec2f screen
      = (vec2f(pixelID)+pixelSample)
      / vec2f(self.fbSize);
    const vec3f origin = self.camera.origin // + lens_offset
      ;
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


