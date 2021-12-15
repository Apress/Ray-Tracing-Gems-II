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
#include "helpers.h"
#include <optix_device.h>
#include <owl/common/math/random.h>

extern "C" __constant__ LaunchParams optixLaunchParams;

typedef owl::common::LCG<4> Random;

struct PRD {
  Random rng;
  float t_hit;
  vec3f gn, sn;
  vec3f texCoord;
  struct {
    vec3f result;
    float importance;
    int depth;
  } radiance;
  struct {
    vec3f attenuation;
  } shadow;
  int max_depth;
};

// ---------------------------------------------------------
// Parallelogram
// ---------------------------------------------------------

OPTIX_BOUNDS_PROGRAM(Parallelogram)(const void *geomData,
                                    box3f &primBounds,
                                    const int primID)
{
  const ParallelogramGeomData &self = *(const ParallelogramGeomData*)geomData;

  // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
  const vec3f tv1  = self.v1 / dot( self.v1, self.v1 );
  const vec3f tv2  = self.v2 / dot( self.v2, self.v2 );
  const vec3f p00  = self.anchor;
  const vec3f p01  = self.anchor + tv1;
  const vec3f p10  = self.anchor + tv2;
  const vec3f p11  = self.anchor + tv1 + tv2;
  const float  area = length(cross(tv1, tv2));
  
  if(area > 0.0f && !isinf(area)) {
    primBounds.lower = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
    primBounds.upper = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
  } else {
    primBounds.lower = vec3f( 1e20f);
    primBounds.upper = vec3f(-1e20f);
  }
}

OPTIX_INTERSECT_PROGRAM(Parallelogram)()
{
  const auto &self
    = owl::getProgramData<ParallelogramGeomData>();

  RadianceRay ray;
  ray.origin = optixGetObjectRayOrigin();
  ray.direction = optixGetObjectRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  vec3f n(self.plane);
  float dt = dot(ray.direction, n );
  float t = (self.plane.w - dot(n, ray.origin))/dt;
  if( t > ray.tmin && t < ray.tmax ) {
    vec3f p = ray.origin + ray.direction * t;
    vec3f vi = p - self.anchor;
    float a1 = dot(self.v1, vi);
    if(a1 >= 0 && a1 <= 1){
      float a2 = dot(self.v2, vi);
      if(a2 >= 0 && a2 <= 1){
        if( optixReportIntersection(t,0,*(unsigned*)&a1,*(unsigned*)&a2) ) {}
      }
    }
  }
}

static
__device__ void phongShade( vec3f p_Kd,
                            vec3f p_Ka,
                            vec3f p_Ks,
                            vec3f p_normal,
                            float p_phong_exp,
                            vec3f p_reflectivity )
{
  const auto &self
    = owl::getProgramData<ParallelogramGeomData>();

  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.origin = optixGetWorldRayOrigin();
  ray.direction = optixGetWorldRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  vec3f hit_point = ray.origin + prd.t_hit * ray.direction;
  
  // ambient contribution
  vec3f result = p_Ka * optixLaunchParams.ambient_light_color;

  // compute direct lighting
  unsigned int num_lights = optixLaunchParams.numLights;

  for(int i = 0; i < num_lights; ++i) {
    // set jittered light direction
    BasicLight light = optixLaunchParams.lights[i];
    vec3f L = light.pos - hit_point;

    vec2f sample = square_to_disk(vec2f(prd.rng(),prd.rng()));
    vec3f U, V, W;
    create_onb(L, U, V, W);
    L += 5.0f * (sample.x * U + sample.y * V);

    float Ldist = length(L);
    L = (1.0f / Ldist) * L;

    float nDl = dot( p_normal, L);

    // cast shadow ray
    PRD shadow_prd;
    shadow_prd.shadow.attenuation = vec3f(1.f);
    if(nDl > 0) {
      ShadowRay shadow_ray(hit_point,L,optixLaunchParams.scene_epsilon,Ldist);
      owl::traceRay(/*accel to trace against*/optixLaunchParams.world,
                    /*the ray to trace*/shadow_ray,
                    /*prd*/shadow_prd);
    }

    // If not completely shadowed, light the hit point
    if(fmaxf(shadow_prd.shadow.attenuation) > 0) {
      vec3f Lc = light.color * shadow_prd.shadow.attenuation;

      result += p_Kd * nDl * Lc;

      vec3f H = normalize(L - ray.direction);
      float nDh = dot( p_normal, H );
      if(nDh > 0) {
        float power = pow(nDh, p_phong_exp);
        result += p_Ks * power * Lc;
      }
    }
  }

  if( fmaxf( p_reflectivity ) > 0 ) {

    // ray tree attenuation
    PRD new_prd;             
    vec3f ntsc_luminance = {0.30, 0.59, 0.11}; 
    new_prd.radiance.importance = prd.radiance.importance * dot( p_reflectivity, ntsc_luminance );
    new_prd.radiance.depth = prd.radiance.depth + 1;

    // reflection ray
    if( new_prd.radiance.importance >= 0.01f && new_prd.radiance.depth <= prd.max_depth) {
      vec3f R = reflect( ray.direction, p_normal );

      RadianceRay refl_ray(hit_point,R,optixLaunchParams.scene_epsilon,1e30f);
      owl::traceRay(/*accel to trace against*/optixLaunchParams.world,
                    /*the ray to trace*/refl_ray,
                    /*prd*/new_prd,
                    /*only CH*/OPTIX_RAY_FLAG_DISABLE_ANYHIT);
      result += p_reflectivity * new_prd.radiance.result;
    }
  }

  // pass the color back up the tree
  prd.radiance.result = result;
}

OPTIX_CLOSEST_HIT_PROGRAM(Parallelogram)()
{
  const auto &self
    = owl::getProgramData<ParallelogramGeomData>();

  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.origin = optixGetWorldRayOrigin();
  ray.direction = optixGetWorldRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  unsigned attr0 = optixGetAttribute_0();
  unsigned attr1 = optixGetAttribute_1();

  float a1 = *(float*)&attr0;
  float a2 = *(float*)&attr1;
  prd.t_hit = optixGetRayTmax();
  prd.sn = prd.gn = vec3f(self.plane);
  prd.texCoord = vec3f(a1,a2,0);

  vec3f uvw = prd.texCoord; // testing

  float4 sampKa, sampKd, sampKs;
  tex2D(&sampKa,self.ka_map,uvw.x,uvw.y);
  tex2D(&sampKd,self.kd_map,uvw.x,uvw.y);
  tex2D(&sampKs,self.ks_map,uvw.x,uvw.y);

  vec3f ka = self.material.Ka * vec3f( sampKa );
  vec3f kd = self.material.Kd * vec3f( sampKd );
  vec3f ks = self.material.Ks * vec3f( sampKs );

  vec3f world_shading_normal = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.sn));
  vec3f world_geometric_normal = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.gn));
  vec3f ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  phongShade( kd, ka, ks, ffnormal, self.material.phong_exp, self.material.reflectivity );
}

OPTIX_ANY_HIT_PROGRAM(Parallelogram)()
{
  PRD &prd = owl::getPRD<PRD>();

  // this material is opaque, so it fully attenuates all shadow rays
  prd.shadow.attenuation = vec3f(0.f);
  
  optixTerminateRay();
}


// ---------------------------------------------------------
// PoolBalls
// ---------------------------------------------------------

OPTIX_BOUNDS_PROGRAM(PoolBall)(const void *geomData,
                               box3f &primBounds,
                               const int primID)
{
  const PoolBallsGeomData &self = *(const PoolBallsGeomData*)geomData;

  const vec3f cen( self.center[primID] );
  const vec3f rad( self.radius );
  
  if( rad.x > 0.0f && !isinf(rad.x) ) {
    primBounds.lower = cen - rad;
    primBounds.upper = cen + rad;
  } else {
    primBounds.lower = vec3f( 1e20f);
    primBounds.upper = vec3f(-1e20f);
  }
}

OPTIX_INTERSECT_PROGRAM(PoolBall)()
{
  const int primID = optixGetPrimitiveIndex();

  const auto &self
    = owl::getProgramData<PoolBallsGeomData>();

  RadianceRay ray;
  ray.origin = optixGetObjectRayOrigin();
  ray.direction = optixGetObjectRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  vec3f center = self.center[primID];
  vec3f O = ray.origin - center;
  vec3f D = ray.direction;
  float radius = self.radius;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);
    bool check_second = true;
    if( optixReportIntersection(root1,0) ) {
      check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc);
      if( optixReportIntersection(root2,0) ) {}
    }
  }
}

OPTIX_CLOSEST_HIT_PROGRAM(PoolBall)()
{
  const int primID = optixGetPrimitiveIndex();

  const auto &self
    = owl::getProgramData<PoolBallsGeomData>();

  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.origin = optixGetWorldRayOrigin();
  ray.direction = optixGetWorldRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  vec3f center = self.center[primID];
  vec3f O = ray.origin - center;
  vec3f D = ray.direction;
  float radius = self.radius;
  linear3f rotation = self.rotation[primID];

  prd.t_hit = optixGetRayTmax();
  prd.sn = prd.gn = (O + prd.t_hit*D)/radius;

  vec3f polar;
  polar.x = dot(rotation.vx, prd.gn);
  polar.y = dot(rotation.vy, prd.gn);
  polar.z = dot(rotation.vz, prd.gn);
  polar = cart_to_pol(polar);

  prd.texCoord = vec3f( polar.x*0.5f*M_1_PIf, (polar.y+M_PI_2f)*M_1_PIf, polar.z/radius );

  // intersection vectors
  const vec3f hit = ray.origin + prd.t_hit * ray.direction;            // hitpoint
  const vec3f N   = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.sn)); // normal
  const vec3f I   = ray.direction;                                            // incident direction
        vec3f R   = reflect(I, N);                                            // reflection direction

  float depth = prd.radiance.depth;  

  float reflection = fresnel_schlick(-dot(N, I), self.material.fresnel_exponent, self.material.fresnel_minimum, self.material.fresnel_maximum);

  // we need not clamp this subtraction because after fresnel_schlick,
  // reflection is guaranteed to be <= fresnel_maximum
  float oneMinusFresnel = self.material.fresnel_maximum - reflection;

  // ambient
  float4 samp;
  if (self.kd_map[primID]) {
    tex2D(&samp, self.kd_map[primID], prd.texCoord.x, prd.texCoord.y);
  } else {
    samp.x=samp.y=samp.z=1.f;
  }
  vec3f kd = self.material.Kd[primID] * vec3f( samp );
  vec3f result = oneMinusFresnel * self.material.Ka * optixLaunchParams.ambient_light_color * kd;

  // direct lighting
  for (unsigned i=0; i<optixLaunchParams.numLights; ++i)
  {
    vec3f L = normalize(optixLaunchParams.lights[i].pos - hit);

    // diffuse
    vec3f diffuse = 1.0f/optixLaunchParams.numLights * ( max(dot(N, L), 0.0f) * optixLaunchParams.lights[i].color );
    result += oneMinusFresnel * diffuse * kd;

    // specular
    result += powf(max(dot(R, L), 0.0f) , self.material.exponent) * self.material.Ks;
  }

  // reflection
  // if (depth < min(self.material.reflection_max_depth, prd.max_depth))
  if (depth < self.material.reflection_max_depth)
  {
    // phong lobe jittering
    vec3f U, V, W;
    create_onb(R, U, V, W);
    R = sample_phong_lobe(vec2f(prd.rng(), prd.rng()), 4096.0f, U, V, W);

    // avoid directions below surface
    if (dot(R, N) < 0.01f)
      R = W;

    // shoot reflection ray
    vec3f ntsc_luminance = {0.30f, 0.59f, 0.11f}; 
    float importance = prd.radiance.importance * reflection * dot( self.material.reflection_color, ntsc_luminance );
    vec3f color = self.material.cutoff_color;
    if ( importance > self.material.importance_cutoff ) {
      PRD new_prd;
      new_prd.t_hit = 1e20f;
      new_prd.radiance.depth = depth+1;
      new_prd.radiance.importance = importance;
      RadianceRay refl_ray(hit,R,optixLaunchParams.scene_epsilon,1e30f);
      owl::traceRay(/*accel to trace against*/optixLaunchParams.world,
                    /*the ray to trace*/refl_ray,
                    /*prd*/new_prd,
                    /*only CH*/OPTIX_RAY_FLAG_DISABLE_ANYHIT);
      color = new_prd.radiance.result;
    }
    result += reflection * self.material.reflection_color * color;
  }

  prd.radiance.result = result;
}

OPTIX_ANY_HIT_PROGRAM(PoolBall)()
{
  PRD &prd = owl::getPRD<PRD>();

  prd.shadow.attenuation = vec3f(0.f);
  optixTerminateRay();
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const auto &lp = optixLaunchParams;
  const vec2i launchIndex = owl::getLaunchIndex();
  const int pixelID = launchIndex.x+self.fbSize.x*launchIndex.y; 

  Random rng(pixelID,lp.accumID);
  
  const vec2f screen = (vec2f(launchIndex)+vec2f(rng(),rng())) / vec2f(self.fbSize);
  RadianceRay ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);
  vec3f ray_target = ray.origin + self.camera.focal_scale * ray.direction;
  // lens sampling
  vec2f sample = square_to_disk(make_float2(rng(), rng()));
  ray.origin = ray.origin + self.camera.aperture_radius * ( sample.x * normalize( self.camera.dir_du ) +  sample.y * normalize( self.camera.dir_dv ) );
  ray.direction = normalize(ray_target - ray.origin);

  //ray.time = 0.5f;

  vec4f accumColor = 0.f;

  PRD prd;
  prd.t_hit = 1e20f;
  prd.radiance.importance = 1.f;
  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/prd,
                /*only CH*/OPTIX_RAY_FLAG_DISABLE_ANYHIT);

  accumColor += vec4f(prd.radiance.result,1.f);

  if (lp.accumID > 0)
    accumColor += vec4f(lp.accumBuffer[pixelID]);
  lp.accumBuffer[pixelID] = accumColor;
  accumColor *= (1.f/(lp.accumID+1));
  self.fbPtr[pixelID]
    = owl::make_rgba(vec3f(accumColor));
}

OPTIX_MISS_PROGRAM(miss)()
{
  const MissProgData &self = owl::getProgramData<MissProgData>();

  PRD &prd = owl::getPRD<PRD>();
  prd.radiance.result = self.bg_color;
}

