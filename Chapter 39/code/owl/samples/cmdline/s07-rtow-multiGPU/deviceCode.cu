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

OPTIX_BOUNDS_PROGRAM(MetalSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<MetalSpheresGeom>(geomData,primBounds,primID); }

OPTIX_BOUNDS_PROGRAM(LambertianSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<LambertianSpheresGeom>(geomData,primBounds,primID); }

OPTIX_BOUNDS_PROGRAM(DielectricSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<DielectricSpheresGeom>(geomData,primBounds,primID); }


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
  
  const vec3f org  = optixGetWorldRayOrigin();
  const vec3f dir  = optixGetWorldRayDirection();
  float hit_t      = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

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
}

OPTIX_INTERSECT_PROGRAM(MetalSpheres)()
{ intersectProg<MetalSpheresGeom>(); }

OPTIX_INTERSECT_PROGRAM(LambertianSpheres)()
{ intersectProg<LambertianSpheresGeom>(); }

OPTIX_INTERSECT_PROGRAM(DielectricSpheres)()
{ intersectProg<DielectricSpheresGeom>(); }


// ==================================================================
// plumbing for closest hit, templated over geometry type so we can
// re-use the same code for different materials
// ==================================================================

// ----------- sphere+material -----------
template<typename SpheresGeomType>
inline __device__
void closestHitSpheres()
{
  const int primID = optixGetPrimitiveIndex();
  const auto &self
    = owl::getProgramData<SpheresGeomType>().prims[primID];
  
  PerRayData &prd = owl::getPRD<PerRayData>();

  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f N     = (hit_P-self.sphere.center);

  prd.out.scatterEvent
    = scatter(self.material,
              hit_P,N,
              prd)
    ? rayGotBounced
    : rayGotCancelled;
}

// ----------- "box+material" -----------
template<typename BoxesGeomType>
inline __device__
void closestHitBoxes()
{
  const auto &self
    = owl::getProgramData<BoxesGeomType>();
  PerRayData &prd = owl::getPRD<PerRayData>();

  // ID of the triangle we've hit:
  const int primID = optixGetPrimitiveIndex();
  
  // there's 12 tris per box:
  const int materialID = primID / 12;
  const auto &material
    = self.perBoxMaterial[materialID];

  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f N      = normalize(cross(B-A,C-A));

  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;

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

// ---------------------- spheres ----------------------
OPTIX_CLOSEST_HIT_PROGRAM(MetalSpheres)()
{ closestHitSpheres<MetalSpheresGeom>(); }
OPTIX_CLOSEST_HIT_PROGRAM(LambertianSpheres)()
{ closestHitSpheres<LambertianSpheresGeom>(); }
OPTIX_CLOSEST_HIT_PROGRAM(DielectricSpheres)()
{ closestHitSpheres<DielectricSpheresGeom>(); }

// ---------------------- boxes ----------------------
OPTIX_CLOSEST_HIT_PROGRAM(MetalBoxes)()
{ closestHitBoxes<MetalBoxesGeom>(); }
OPTIX_CLOSEST_HIT_PROGRAM(LambertianBoxes)()
{ closestHitBoxes<LambertianBoxesGeom>(); }
OPTIX_CLOSEST_HIT_PROGRAM(DielectricBoxes)()
{ closestHitBoxes<DielectricBoxesGeom>(); }





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
  /* nothing to do */
}



inline __device__
vec3f tracePath(const RayGenData &self,
                owl::Ray &ray, PerRayData &prd)
{
  vec3f attenuation = 1.f;
  
  /* iterative version of recursion, up to depth 50 */
  for (int depth=0;depth<50;depth++) {
    prd.out.scatterEvent = rayDidntHitAnything;
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

  /* compute who is repsonible for a given group of pixels: */
  int deviceThatIsResponsible = (pixelID.x>>5) % self.deviceCount;
  /* ... and if it's not us, just return (some other device will
     compute these pixels */
  if (self.deviceIndex != deviceThatIsResponsible)
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


