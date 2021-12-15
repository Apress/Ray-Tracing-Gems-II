// ======================================================================== //
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
#include "constants.h"
#include <optix_device.h>

#include <owl/common/math/random.h>
#include <owl/common/math/LinearSpace.h>


__constant__ LaunchParams optixLaunchParams;


typedef owl::common::LCG<4> Random;
typedef owl::RayT<0, 3> RadianceRay;
typedef owl::RayT<1, 3> ShadowRay;
typedef owl::RayT<2, 3> OutlineShadowRay;

inline __device__
vec3f missColor(const RadianceRay &ray)
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

OPTIX_MISS_PROGRAM(miss_shadow)()
{
  float &vis = owl::getPRD<float>();
  vis = 1.f;
}

typedef enum {
  /*! ray could get properly bounced, and is still alive */
  rayGotBounced,
  /*! ray didn't hit anything, and went into the environment */
  rayDidntHitAnything
} ScatterEvent;


struct PerRayData
{
  Random random;
  struct {
    ScatterEvent scatterEvent;
    vec3f        scattered_origin;
    vec3f        scattered_direction;
    vec3f        attenuation;
    vec3f        directLight;
    float        hitDistance;
  } out;
};

template <typename RayT, int Mask>
inline __device__
RayT makeRay(const vec3f &origin,
             const vec3f &direction,
             float tmin,
             float tmax)
{
  if (optixLaunchParams.enableClipping) {
    const float eps = 0.01f * optixLaunchParams.brickScale;
    const float clipZ = optixLaunchParams.clipHeight * optixLaunchParams.brickScale;
    const float t = (clipZ - origin.z) / direction.z;
    if (direction.z < 0.f) {
      tmin = owl::max(tmin, t-eps);
    } else {
      tmax = owl::min(tmax, t+eps);
    }
  }

  return RayT(origin, direction, tmin, tmax, Mask);
}



inline __device__
RadianceRay makeRadianceRay(const vec3f &origin, 
                            const vec3f &direction,
                            float tmin,
                            float tmax)
{
  return makeRay<RadianceRay, VISIBILITY_RADIANCE>(origin, direction, tmin, tmax);
}

inline __device__
ShadowRay makeShadowRay(const vec3f &origin, 
                        const vec3f &direction,
                        float tmin,
                        float tmax)
{
  return makeRay<ShadowRay, VISIBILITY_SHADOW>(origin, direction, tmin, tmax);
}

inline __device__
OutlineShadowRay makeOutlineShadowRay(const vec3f &origin, 
                                      const vec3f &direction,
                                      float tmin,
                                      float tmax)
{
  return makeRay<OutlineShadowRay, VISIBILITY_OUTLINE>(origin, direction, tmin, tmax);
}

inline __device__
vec3f tracePrimaryRay(const RayGenData &self,
                RadianceRay &ray, PerRayData &prd)
{
  if (optixLaunchParams.enableToonOutline) {
    prd.out.hitDistance = 1e10f;
  }
  
  prd.out.scatterEvent = rayDidntHitAnything;
  owl::traceRay(optixLaunchParams.world,
                ray,
                prd,
                OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
    
  if (prd.out.scatterEvent == rayDidntHitAnything) {
    return missColor(ray);
  }
  else { // ray is still alive, and got properly bounced
    ray = makeRadianceRay(prd.out.scattered_origin,
                          prd.out.scattered_direction,
                          0.f, // rely on hitP offset along normal
                          1e10f);
  }
  return prd.out.directLight;
}

inline __device__
vec3f traceBounces(const RayGenData &self,
                RadianceRay &ray, PerRayData &prd)
{
  vec3f attenuation = prd.out.attenuation;
  vec3f directLight = 0.f;

  constexpr int MaxDepth = 2;
  
  /* iterative version of recursion, up to max depth */
  for (int depth=1;depth<MaxDepth;depth++) {
    prd.out.scatterEvent = rayDidntHitAnything;
    owl::traceRay(/*accel to trace against*/ optixLaunchParams.world,
                  /*the ray to trace*/ ray,
                  /*prd*/prd,
                  OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
    
    if (prd.out.scatterEvent == rayDidntHitAnything)
      /* ray got 'lost' to the environment - 'light' it with miss
         shader */
      return directLight + attenuation * missColor(ray);
    else { // ray is still alive, and got properly bounced
      attenuation *= prd.out.attenuation;
      directLight += prd.out.directLight;
      ray = makeRadianceRay(/* origin   : */ prd.out.scattered_origin,
                     /* direction: */ prd.out.scattered_direction,
                     /* tmin     : */ 0.f,  // rely on hitP offset along normal
                     /* tmax     : */ 1e10f);

    }
  }
  // recursion did not terminate - cancel it but return direct lighting from any previous bounce
  return directLight;
}

// returns a visibility term (1 for unshadowed)
inline __device__
float traceShadowRay(const OptixTraversableHandle &traversable,
                  ShadowRay &ray)
{
  float vis = 0.f;
  owl::traceRay(traversable, ray, vis, 
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT
                   | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                   | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return vis;
}

// returns a visibility term (1 for unshadowed)
inline __device__
float traceOutlineShadowRay(const OptixTraversableHandle &traversable,
                  OutlineShadowRay &ray)
{
  float vis = 0.f;
  owl::traceRay(traversable, ray, vis, 
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT
                   | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
                   | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                   | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return vis;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  const vec2i fbSize = optixLaunchParams.fbSize;
  const int fbIndex = pixelID.x+fbSize.x*pixelID.y;

  PerRayData prd;
  prd.random.init(fbIndex, optixLaunchParams.frameID);

  // Note: measured to be faster to keep the loop over subpixels at the launch level,
  // not inside raygen.  Maybe a long tail effect?
  constexpr int NUM_SAMPLES_PER_PIXEL = 1;
  vec3f accumColor = 0.f;

  for (int sampleID=0;sampleID<NUM_SAMPLES_PER_PIXEL;sampleID++) {

    const vec2f pixelSample(prd.random(), prd.random());
    const vec2f screen = (vec2f(pixelID)+pixelSample) / vec2f(fbSize);

    const vec3f rayDir = normalize(self.camera.dir_00
                  + screen.u * self.camera.dir_du
                  + screen.v * self.camera.dir_dv);

    RadianceRay ray = makeRadianceRay(self.camera.pos, rayDir, 0.f, 1e30f);

    vec3f color = tracePrimaryRay(self, ray, prd);

    float visibility = 1.f;
    if (optixLaunchParams.enableToonOutline) {

      // Feature size control for outlines
      const float outlineDepthBias = 5*optixLaunchParams.brickScale;
      const float firstHitDistance = prd.out.hitDistance; // Note: dependency on primary ray
      OutlineShadowRay outlineShadowRay = makeOutlineShadowRay(self.camera.pos,
          rayDir, 0.f, firstHitDistance-outlineDepthBias);
      visibility = traceOutlineShadowRay(optixLaunchParams.world, outlineShadowRay);
    }
    // Note: measurable speedup from tracing bounce rays unconditionally without
    // checking visibility first.
    if (prd.out.scatterEvent == rayGotBounced) {
      color += traceBounces(self, ray, prd);
    } 
    accumColor += color*visibility;
  }
    
  vec4f rgba {accumColor / NUM_SAMPLES_PER_PIXEL, 1.0f};

  if (optixLaunchParams.frameID > 0) {
    // Blend with accum buffer
    const vec4f accum = optixLaunchParams.fbAccumBuffer[fbIndex];
    rgba += float(optixLaunchParams.frameID) * accum; 
    rgba /= (optixLaunchParams.frameID+1.f);
  }

  optixLaunchParams.fbAccumBuffer[fbIndex] = (float4)rgba;
  optixLaunchParams.fbPtr[fbIndex] = owl::make_rgba(rgba);
}

inline __device__ 
float2 squareToDisk(float u1, float u2)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  float2 p = {r * cosf( phi ), r * sinf( phi )};
  return p;
}

inline __device__
vec3f cosineSampleHemisphere(float u1, float u2)
{
  float2 p = squareToDisk(u1, u2);

  // Project up to hemisphere.
  return vec3f(p.x, p.y, sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) ));
}

inline __device__
vec3f sunlight(const vec3f &hit_P_offset, const vec3f &Ng, Random &random)
{
  const vec3f lightDir = optixLaunchParams.sunDirection;

  const float NdotL = dot(Ng, lightDir);
  if (NdotL <= 0.f) {
    return 0.f;  // below horizon
  }

  // Build frame around light dir
  const owl::LinearSpace3f lightFrame = owl::common::frame(normalize(lightDir));

  // jitter light direction slightly
  const float lightRadius = 0.01f;  // should be ok as a constant since our scenes are normalized
  const vec3f lightCenter = hit_P_offset + lightFrame.vz;
  const float2 sample = squareToDisk(random(), random());
  const vec3f jitteredPos = lightCenter + lightRadius*(sample.x*lightFrame.vx + sample.y*lightFrame.vy);
  const vec3f jitteredLightDir = jitteredPos - hit_P_offset;  // no need to normalize

  ShadowRay shadowRay = makeShadowRay(hit_P_offset, jitteredLightDir, 0.f, 1e10f);
  float vis = traceShadowRay(optixLaunchParams.world, shadowRay);
  return vis * optixLaunchParams.sunColor * NdotL; 

}

inline __device__ 
vec3f scatterLambertian(const vec3f &Ng, Random &random)
{
  const owl::LinearSpace3f shadingFrame = owl::common::frame(Ng);
  vec3f scatteredDirectionInShadingFrame = cosineSampleHemisphere(random(), random());
  vec3f scatteredDirection = shadingFrame.vx * scatteredDirectionInShadingFrame.x +
                             shadingFrame.vy * scatteredDirectionInShadingFrame.y +
                             shadingFrame.vz * scatteredDirectionInShadingFrame.z;

  return scatteredDirection;
}

// Common reflectance function for all geometry
inline __device__
void shade(const vec3f &Ng, const vec3f &color)
{
  PerRayData &prd = owl::getPRD<PerRayData>();
  const vec3f dir   = optixGetWorldRayDirection();
  const vec3f org   = optixGetWorldRayOrigin();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const float shadowBias = 0.01f * fminf(1.f, optixLaunchParams.brickScale);
  const vec3f hit_P_offset = hit_P + shadowBias*Ng;  // bias along normal to help with shadow acne

  // Direct
  const vec3f directLight = sunlight(hit_P_offset, Ng, prd.random);

  // Bounce
  vec3f scatteredDirection = scatterLambertian(Ng, prd.random);

  prd.out.directLight = directLight*color;
  prd.out.attenuation = color;
  prd.out.scatterEvent = rayGotBounced;
  prd.out.scattered_direction = scatteredDirection;
  prd.out.scattered_origin = hit_P_offset;

  if (optixLaunchParams.enableToonOutline) {
    prd.out.hitDistance = length(hit_P - org);
  }
}

inline __device__
void shadeTriangleOnBrick(unsigned int brickID)
{
  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  // Compute normal for triangle
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f Nbox   = normalize(cross(B-A,C-A));
  const vec3f Ng     = normalize(vec3f(optixTransformNormalFromObjectToWorldSpace(Nbox)));

  // Get brick color
  const int ci = self.colorIndexPerBrick[brickID];
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  shade(Ng, color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  unsigned int brickID = optixGetPrimitiveIndex() / self.primCountPerBrick;
  shadeTriangleOnBrick(brickID);
}

OPTIX_CLOSEST_HIT_PROGRAM(InstancedTriangleMesh)()
{
  unsigned int brickID = optixGetInstanceId();
  shadeTriangleOnBrick(brickID);
}

OPTIX_BOUNDS_PROGRAM(VoxGeom)(const void *geomData,
                              box3f &primBounds,
                              const int primID)
{
  const VoxGeomData &self = *(const VoxGeomData*)geomData;
  uchar4 indices = self.prims[primID];
  vec3f boxmin( indices.x, indices.y, indices.z );
  vec3f boxmax( 1+indices.x, 1+indices.y, 1+indices.z );

  if (self.enableToonOutline) {
    // bloat the box slightly
    const vec3f boxcenter (indices.x + 0.5f, indices.y + 0.5f, indices.z + 0.5f);
    boxmin = boxcenter + OUTLINE_SCALE*(boxmin-boxcenter);
    boxmax = boxcenter + OUTLINE_SCALE*(boxmax-boxcenter);
  }
  
  primBounds = box3f(boxmin, boxmax);
}



namespace {
  // temp swizzles
  inline __device__
  vec2f yz(vec3f v)
  {
    return vec2f(v.y, v.z);
  }
  inline __device__
  vec2f zx(vec3f v)
  {
    return vec2f(v.z, v.x);
  }
  inline __device__
  vec2f xy(vec3f v)
  {
    return vec2f(v.x, v.y);
  }
  inline __device__
  vec3f yzx(vec3f v)
  {
    return vec3f(v.y, v.z, v.x);
  }
  inline __device__
  vec3i zxy(vec3i v)
  {
    return vec3i(v.z, v.x, v.y);
  }
}

// Ray-box intersection with normals from Majercik et al 2018
OPTIX_INTERSECT_PROGRAM(VoxGeomMajercik)()
{
  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();
  uchar4 indices = self.prims[primID];
  vec3f boxCenter(indices.x+0.5, indices.y+0.5, indices.z+0.5);

  // Translate ray to local box space
  const vec3f rayOrigin  = vec3f(optixGetObjectRayOrigin()) - boxCenter;
  const vec3f rayDirection  = optixGetObjectRayDirection();  // assume no rotation
  const vec3f invRayDirection = vec3f(1.0f) / rayDirection;

  // Negated sign function
  const vec3f sgn( 
      rayDirection.x > 0.f ? -1 : 1,
      rayDirection.y > 0.f ? -1 : 1,
      rayDirection.z > 0.f ? -1 : 1);

  const vec3f boxRadius(0.5f, 0.5f, 0.5f);
  vec3f distanceToPlane = boxRadius*sgn - rayOrigin;
  distanceToPlane *= invRayDirection;

  const bool testX = distanceToPlane.x >= 0.f && 
    owl::all_less_than(owl::abs(yz(rayOrigin) + yz(rayDirection)*distanceToPlane.x), yz(boxRadius));

  const bool testY = distanceToPlane.y >= 0.f &&
    owl::all_less_than(owl::abs(zx(rayOrigin) + zx(rayDirection)*distanceToPlane.y), zx(boxRadius));

  const bool testZ = distanceToPlane.z >= 0.f &&
    owl::all_less_than(owl::abs(xy(rayOrigin) + xy(rayDirection)*distanceToPlane.z), xy(boxRadius));

  const vec3b test(testX, testY, testZ);
  if ( test.x || test.y || test.z ) { // hit the box
    float distance = test.x ? distanceToPlane.x : (test.y ? distanceToPlane.y : distanceToPlane.z);
    const float ray_tmax = optixGetRayTmax();
    const float ray_tmin = optixGetRayTmin();
    if (distance > ray_tmin && distance < ray_tmax) {  // closer than existing hit
      // Since N is something like [0,-1,0], encode it as sign (1 bit) and 3 components (3 bits): 000...SNNN
      // This lets it fit in one attribute.
      int signOfN = (sgn.x*test.x + sgn.y*test.y + sgn.z*test.z) > 0 ? 1 : 0;
      int packedN = (signOfN << 3) | (test.z << 2) | (test.y << 1) | test.x;
      optixReportIntersection(distance, 0, packedN);
    }
  }
}

// Use the "efficient slabs" method for shadow rays where we don't need normals.
template <bool IsOutline>
inline __device__ 
void intersectVoxGeomShadow()
{
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();

  // Per-brick outline control
  if (IsOutline && !self.enableToonOutline) {
      return;
  }

  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  uchar4 indices = self.prims[primID];
  vec3f boxCenter(indices.x+0.5, indices.y+0.5, indices.z+0.5);

  // Translate ray to local box space
  const vec3f rayOrigin  = vec3f(optixGetObjectRayOrigin()) - boxCenter;
  const vec3f rayDirection  = optixGetObjectRayDirection();  // assume no rotation
  const vec3f invRayDirection = vec3f(1.0f) / rayDirection;

  const float ray_tmax = optixGetRayTmax();
  const float ray_tmin = optixGetRayTmin();

  constexpr float radius = IsOutline ? 0.5f*OUTLINE_SCALE : 0.5f;
  const vec3f boxRadius(radius, radius, radius);
  vec3f t0 = (-boxRadius - rayOrigin) * invRayDirection;
  vec3f t1 = ( boxRadius - rayOrigin) * invRayDirection;
  float tnear = reduce_max(owl::min(t0, t1));
  float tfar  = reduce_min(owl::max(t0, t1));
  float distance = IsOutline ? tfar : (tnear > 0.f ? tnear : tfar);

  if (tnear <= tfar && distance > ray_tmin && distance < ray_tmax) {
    optixReportIntersection( distance, 0);
  }
}

OPTIX_INTERSECT_PROGRAM(VoxGeomShadow)()
{
  return intersectVoxGeomShadow</*IsOutline=*/ false>();
}

OPTIX_INTERSECT_PROGRAM(VoxGeomOutlineShadow)()
{
  return intersectVoxGeomShadow</*IsOutline=*/ true>();
}

inline __device__ 
vec3f unpackNormal(int packedN)
{
  const int sgnN = (packedN >> 3) ? 1 : -1;
  const float3 Nbox = make_float3(
    sgnN * ( packedN       & 1),
    sgnN * ((packedN >> 1) & 1),
    sgnN * ((packedN >> 2) & 1));
  return Nbox;
}

OPTIX_CLOSEST_HIT_PROGRAM(VoxGeom)()
{
  // Get normal for whichever face we hit
  const int packedN = optixGetAttribute_0();
  const vec3f Nbox = unpackNormal(packedN);
  const vec3f Ng = normalize(vec3f(optixTransformNormalFromObjectToWorldSpace(Nbox)));

  // Get brick color
  const int primID = optixGetPrimitiveIndex();
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();
  const int ci = self.prims[primID].w;
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  shade(Ng, color);
}


// Experiment with "blocks" (small grids) of bricks

OPTIX_BOUNDS_PROGRAM(VoxBlockGeom)(const void *geomData,
                              box3f &primBounds,
                              const int primID)
{
  const VoxBlockGeomData &self = *(const VoxBlockGeomData*)geomData;
  uchar3 indices = self.prims[primID];
  vec3f boxmin( indices.x, indices.y, indices.z );
  vec3f boxmax = boxmin + vec3f(BLOCKLEN);

  // Not obvious how to do toon outline for blocks
  /*
  if (self.enableToonOutline) {
    ... 
  }
  */
  
  primBounds = box3f(boxmin, boxmax);
}

__device__ inline
bool intersectRayBox(const vec3f _rayOrigin, const vec3f rayDirection,
    vec3f boxCenter, vec3f boxRadius,
    float &tnear, float &tfar, vec3i &axismask)
{
  // Translate ray to local box space
  const vec3f rayOrigin  = _rayOrigin - boxCenter;
  const vec3f invRayDirection = vec3f(1.0f) / rayDirection;

  vec3f t0 = (-boxRadius - rayOrigin) * invRayDirection;
  vec3f t1 = ( boxRadius - rayOrigin) * invRayDirection;
  vec3f tnearPerAxis = owl::min(t0, t1);
  tnear = reduce_max(tnearPerAxis);
  tfar  = reduce_min(owl::max(t0, t1));
  axismask = vec3i(owl::ge(tnearPerAxis, vec3f(tnear)));
  return tnear <= tfar;
}

template <bool IsShadowRay>
inline __device__ 
void intersectVoxBlockGeom()
{
  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  const VoxBlockGeomData &self = owl::getProgramData<VoxBlockGeomData>();

  const uchar3 blockOrigin = self.prims[primID];
  const vec3f blockRadius(0.5f*BLOCKLEN);
  const vec3f blockCenter = vec3f(blockOrigin.x, blockOrigin.y, blockOrigin.z) + blockRadius;

  const vec3f rayOrigin = optixGetObjectRayOrigin();
  const vec3f rayDirection = optixGetObjectRayDirection();
  const float rayTmin = optixGetRayTmin();
  const float rayTmax = optixGetRayTmax();

  const vec3f blockOrigin3f(blockOrigin.x, blockOrigin.y, blockOrigin.z);

  // Intersect ray with block
  float tnear;       // initial t as ray enters block, updated during DDA
  vec3i axismask(0); // initial axis mask (==abs(N)), updated during DDA
  float tfar;
  if (!intersectRayBox(rayOrigin, rayDirection, blockCenter, blockRadius, tnear, tfar, axismask)) {
    return;  // Ray line misses block (without considering ray span)
  }

  tnear = max(tnear, rayTmin);
  tfar = min(tfar, rayTmax);
  if (tnear > tfar) return;

  
  // Set up DDA for the block grid
  
  // Constants during traversal

  const vec3i blockDim(BLOCKLEN);

  const vec3f sgn( 
      rayDirection.x > 0.f ? 1 : -1,
      rayDirection.y > 0.f ? 1 : -1,
      rayDirection.z > 0.f ? 1 : -1);

  const vec3f invRayDirection = vec3f(1.0f) / rayDirection;

  const vec3f deltaT = sgn * invRayDirection;
  const vec3i step (sgn.x, sgn.y, sgn.z);
  const vec3i exitCell ( 
      sgn.x < 0 ? -1 : blockDim.x,
      sgn.y < 0 ? -1 : blockDim.y,
      sgn.z < 0 ? -1 : blockDim.z);

  const int brickOffset = primID*blockDim.x*blockDim.y*blockDim.z;


  // Things that change during traversal

  // Current cell in grid
  const vec3f pos = rayOrigin + tnear*rayDirection - blockOrigin3f;
  vec3i cell = owl::clamp(vec3i(pos), vec3i(0), blockDim-vec3i(1));

  // T values where the ray crosses into the next "slab" along the x,y,z axes
  // Version from: https://www.shadertoy.com/view/MdBGRm
  const vec3f corner = owl::max(sgn, vec3f(0.f));
  vec3f nextCrossingT = vec3f(tnear) + (vec3f(cell.x, cell.y, cell.z) + corner - pos)  * invRayDirection;

  // Color at entry cell, provided the ray started outside the block.  Updated during DDA.
  // Note: we ignore the case where a ray starts inside an opaque cell (probably due to shadow bias).
  int brickIdx = brickOffset + cell.x + cell.y*blockDim.x + cell.z*blockDim.x*blockDim.y;
  int colorIdx = tnear > rayTmin ? self.colorIndices[brickIdx] : -1;

  // DDA traversal
  // Note: profiling shows small gain from using a fixed size loop here even though we always break.
  // Theoretical length of diagonal: blocklen*sqrt(3).  Had to increase this empirically.
  constexpr int MaxNumSteps = BLOCKLEN*3;
  for (int i = 0; i < MaxNumSteps; ++i) {
    
    if (colorIdx > 0) {
      // previous t-crossing is the intersection t
      const vec3f ct = nextCrossingT - deltaT;
      tnear = owl::reduce_max(ct);
      //if (tnear >= rayTmax) break;  // Can be skipped for our scenes.
      if (IsShadowRay) {
        optixReportIntersection(tnear, 0);
      } else {
        int signOfNormal = owl::any(owl::lt(sgn*vec3f(axismask), vec3f(0.f)));
        int packedNormal = (signOfNormal << 3) | (axismask.z << 2) | (axismask.y << 1) | axismask.x;
        optixReportIntersection(tnear, 0, packedNormal, colorIdx);
      }
      break;
    }

    // Advance to next cell along ray
    // Version from: https://www.shadertoy.com/view/MdBGRm
    vec3i cp = vec3i(owl::lt(nextCrossingT, yzx(nextCrossingT)));
    axismask = cp * (vec3i(1) - zxy(cp));

    cell += step * axismask;
    if (cell*axismask == exitCell*axismask) {
      break;
    } 
    nextCrossingT += deltaT*vec3f(axismask);

    brickIdx = brickOffset + cell.x + cell.y*blockDim.x + cell.z*blockDim.x*blockDim.y;
    colorIdx = self.colorIndices[brickIdx];
  }

}

OPTIX_INTERSECT_PROGRAM(VoxBlockGeom)()
{
  intersectVoxBlockGeom</*IsShadow=*/ false>();
}

OPTIX_INTERSECT_PROGRAM(VoxBlockGeomShadow)()
{
  intersectVoxBlockGeom</*IsShadow=*/ true>();
}

OPTIX_CLOSEST_HIT_PROGRAM(VoxBlockGeom)()
{
  // Get normal for whichever face we hit
  const int packedN = optixGetAttribute_0();
  const vec3f Nbox = unpackNormal(packedN);
  const vec3f Ng = normalize(vec3f(optixTransformNormalFromObjectToWorldSpace(Nbox)));

  // Get brick color
  const VoxBlockGeomData &self = owl::getProgramData<VoxBlockGeomData>();
  const int ci = optixGetAttribute_1();
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  shade(Ng, color);
}

