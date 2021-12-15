// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

// Create and ray trace a solid cube made out of tiny spheres.

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "t01-manySpheres.png";
const vec2i fbSize(2048,2048);
//const vec2i fbSize(1600,800);
const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<LambertianSphere> lambertianSpheres;

// Not needed by default, but here if you want to randomize positions at some point.
//inline float rnd()
//{
//  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
//  static std::uniform_real_distribution<float> dis(0.f, 1.f);
//  return dis(gen);
//}
//
//inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

void createScene(int N)
{
  for (int iz=0;iz<N;iz++)
    for (int iy=0;iy<N;iy++)
      for (int ix=0;ix<N;ix++) {
        Sphere sphere = {vec3f((float)ix,(float)iy,(float)iz), .5f};
        // Sphere sphere = {vec3f(ix,iy+.01f*ix,iz+.1f*ix), .45f};
        lambertianSpheres.push_back({sphere,
                                     Lambertian{vec3f(0.5f, 0.5f, 0.5f)}});
      }
}

int main(int ac, char **av)
{
  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

  LOG("ll example '" << av[0] << "' starting up");

  int N = 500;

  LOG("creating the scene ...");
  createScene(N);
  LOG_OK("created scene:");
  LOG_OK(" num lambertian spheres: " << lambertianSpheres.size());

  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate();
  OWLModule  module  = owlModuleCreate(context,ptxCode);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type(s)
  // -------------------------------------------------------

  // ----------- lambertian -----------
  OWLVarDecl lambertianSpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(LambertianSpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType lambertianSpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(LambertianSpheresGeom),
                        lambertianSpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(lambertianSpheresGeomType,0,
                           module,"LambertianSpheres");
  owlGeomTypeSetIntersectProg(lambertianSpheresGeomType,0,
                              module,"LambertianSpheres");
  owlGeomTypeSetBoundsProg(lambertianSpheresGeomType,
                           module,"LambertianSpheres");
  // make sure to do that *before* setting up the geometry, since the
  // user geometry group will need the compiled bounds programs upon
  // accelBuild()
  owlBuildPrograms(context);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  // ----------- lambertian -----------
  OWLBuffer lambertianSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(lambertianSpheres[0]),
                            lambertianSpheres.size(),lambertianSpheres.data());
  OWLGeom lambertianSpheresGeom
    = owlGeomCreate(context,lambertianSpheresGeomType);
  owlGeomSetPrimCount(lambertianSpheresGeom,lambertianSpheres.size());
  owlGeomSetBuffer(lambertianSpheresGeom,"prims",lambertianSpheresBuffer);

  // // ##################################################################
  // // set up all *ACCELS* we need to trace into those groups
  // // ##################################################################

#if 1
  OWLGroup grp
    = owlUserGeomGroupCreate(context,1,&lambertianSpheresGeom);
    // = owlUserGeomGroupCreate(context,1,userGeoms);
  owlGroupBuildAccel(grp);
  OWLGroup world
    = owlInstanceGroupCreate(context,1);
  owlInstanceGroupSetChild(world,0,grp);
#else
  OWLGroup world
    = owlUserGeomGroupCreate(context,1,&lambertianSpheresGeom);
    // = owlUserGeomGroupCreate(context,1,userGeoms);
#endif
  owlGroupBuildAccel(world);

  // ##################################################################
  // set miss and raygen programs
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[] = {
    { /* sentinel to mark end of list */ }
  };
  // ........... create object  ............................
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  owlMissProgSet(context,0,missProg);

  // ........... set variables  ............................
  /* nothing to set */

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "deviceIndex",   OWL_DEVICE, OWL_OFFSETOF(RayGenData,deviceIndex)},
    { "deviceCount",   OWL_INT,    OWL_OFFSETOF(RayGenData,deviceCount)},
    { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.origin)},
    { "camera.llc",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.lower_left_corner)},
    { "camera.horiz",  OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.horizontal)},
    { "camera.vert",   OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.vertical)},
    { /* sentinel to mark end of list */ }
  };

  // ........... create object  ............................
  OWLRayGen rayGen
    = owlRayGenCreate(context,module,"rayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

  // ........... compute variable values  ..................
  const float vfov = fovy;
  const vec3f vup = lookUp;
  const float aspect = fbSize.x / float(fbSize.y);
  const float theta = vfov * ((float)M_PI) / 180.0f;
  const float half_height = tanf(theta / 2.0f);
  const float half_width = aspect * half_height;
  const float focusDist = 10.f;
  vec3f lookFrom = 1.8f*vec3f(1.3f,1.5f,2.f)*vec3f((float)N);
  vec3f lookAt   = vec3f(0.5f*N);
  const vec3f origin = lookFrom;
  const vec3f w = normalize(lookFrom - lookAt);
  const vec3f u = normalize(cross(vup, w));
  const vec3f v = cross(w, u);
  const vec3f lower_left_corner
    = origin - half_width * focusDist*u - half_height * focusDist*v - focusDist * w;
  const vec3f horizontal = 2.0f*half_width*focusDist*u;
  const vec3f vertical = 2.0f*half_height*focusDist*v;

  // ----------- set variables  ----------------------------
  owlRayGenSet1i    (rayGen,"deviceCount",  owlGetDeviceCount(context));
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSetGroup (rayGen,"world",        world);
  owlRayGenSet3f    (rayGen,"camera.org",   (const owl3f&)origin);
  owlRayGenSet3f    (rayGen,"camera.llc",   (const owl3f&)lower_left_corner);
  owlRayGenSet3f    (rayGen,"camera.horiz", (const owl3f&)horizontal);
  owlRayGenSet3f    (rayGen,"camera.vert",  (const owl3f&)vertical);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  // programs have been built before, but have to rebuild raygen and
  // miss progs
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  LOG("launching ...");
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);

  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb
    = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################

  LOG("destroying devicegroup ...");
  owlContextDestroy(context);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
