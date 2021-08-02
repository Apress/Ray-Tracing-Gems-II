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

// Extension of the rtow-mixedGeometires example that shows how to do
// multi-GPU. See README.md for more details

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "s07-rtow-multiGPU.png";
const vec2i fbSize(1600,800);
const vec3f lookFrom(13, 2, 3);
const vec3f lookAt(0, 0, 0);
const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<DielectricSphere> dielectricSpheres;
std::vector<LambertianSphere> lambertianSpheres;
std::vector<MetalSphere>      metalSpheres;

struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<Dielectric> materials;
} dielectricBoxes;
struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<Metal> materials;
} metalBoxes;
struct {
  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;
  std::vector<Lambertian> materials;
} lambertianBoxes;

inline float rnd()
{
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

inline vec3f randomPointInUnitSphere()
{
  vec3f p;
  do {
    p = 2.f*vec3f(rnd(),rnd(),rnd()) - vec3f(1.f);
  } while (dot(p,p) >= 1.f);
  return p;
}

template<typename BoxArray, typename Material>
void addRandomBox(BoxArray &boxes,
                  const vec3f &center,
                  const float size,
                  const Material &material)
{
  const int NUM_VERTICES = 8;
  static const vec3f unitBoxVertices[NUM_VERTICES] =
    {
      {-1.f, -1.f, -1.f},
      {+1.f, -1.f, -1.f},
      {+1.f, +1.f, -1.f},
      {-1.f, +1.f, -1.f},
      {-1.f, +1.f, +1.f},
      {+1.f, +1.f, +1.f},
      {+1.f, -1.f, +1.f},
      {-1.f, -1.f, +1.f},
    };

  const int NUM_INDICES = 12;
  static const vec3i unitBoxIndices[NUM_INDICES] =
    {
      {0, 2, 1}, //face front
      {0, 3, 2},
      {2, 3, 4}, //face top
      {2, 4, 5},
      {1, 2, 5}, //face right
      {1, 5, 6},
      {0, 7, 4}, //face left
      {0, 4, 3},
      {5, 4, 7}, //face back
      {5, 7, 6},
      {0, 6, 7}, //face bottom
      {0, 1, 6}
    };

  const vec3f U = normalize(randomPointInUnitSphere());
  owl::affine3f xfm = owl::frame(U);
  xfm = owl::affine3f(owl::linear3f::rotate(U,rnd())) * xfm;
  xfm = owl::affine3f(owl::linear3f::scale(.7f*size)) * xfm;
  xfm = owl::affine3f(owl::affine3f::translate(center)) * xfm;
  
  const int startIndex = (int)boxes.vertices.size();
  for (int i=0;i<NUM_VERTICES;i++)
    boxes.vertices.push_back(owl::xfmPoint(xfm,unitBoxVertices[i]));
  for (int i=0;i<NUM_INDICES;i++)
    boxes.indices.push_back(unitBoxIndices[i]+vec3i(startIndex));
  boxes.materials.push_back(material);
}

void createScene()
{
  lambertianSpheres.push_back({Sphere{vec3f(0.f, -1000.0f, -1.f), 1000.f},
        Lambertian{vec3f(0.5f, 0.5f, 0.5f)}});
  
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float choose_shape = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        if (choose_shape > .5f) {
          addRandomBox(lambertianBoxes,center,.2f,
                       Lambertian{rnd3f()*rnd3f()});
        } else
          lambertianSpheres.push_back({Sphere{center, 0.2f},
                Lambertian{rnd3f()*rnd3f()}});
      } else if (choose_mat < 0.95f) {
        if (choose_shape > .5f) {
          addRandomBox(metalBoxes,center,.2f,
                       Metal{0.5f*(1.f+rnd3f()),0.5f*rnd()});
        } else
          metalSpheres.push_back({Sphere{center, 0.2f},
                Metal{0.5f*(1.f+rnd3f()),0.5f*rnd()}});
      } else {
        if (choose_shape > .5f) {
          addRandomBox(dielectricBoxes,center,.2f,
                       Dielectric{1.5f});
        } else
          dielectricSpheres.push_back({Sphere{center, 0.2f},
                Dielectric{1.5f}});
      }
    }
  }
  dielectricSpheres.push_back({Sphere{vec3f(0.f, 1.f, 0.f), 1.f},
        Dielectric{1.5f}});
  lambertianSpheres.push_back({Sphere{vec3f(-4.f,1.f, 0.f), 1.f},
        Lambertian{vec3f(0.4f, 0.2f, 0.1f)}});
  metalSpheres.push_back({Sphere{vec3f(4.f, 1.f, 0.f), 1.f},
        Metal{vec3f(0.7f, 0.6f, 0.5f), 0.0f}});
}
  
int main(int ac, char **av)
{
  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

  LOG("owl example '" << av[0] << "' starting up");

  LOG("creating the scene ...");
  createScene();
  LOG_OK("created scene:");
  LOG_OK(" num lambertian spheres: " << lambertianSpheres.size());
  LOG_OK(" num dielectric spheres: " << dielectricSpheres.size());
  LOG_OK(" num metal spheres     : " << metalSpheres.size());
  
  // ##################################################################
  // init owl
  // ##################################################################

  /*! create a new device, and pass {nullptr,0}, which indicates "all
      GPUs you can find */
  OWLContext context = owlContextCreate(nullptr,0);

  int numGPUsFound = owlGetDeviceCount(context);
  LOG("Context created; owl found " << numGPUsFound << " GPUs");
  OWLModule  module  = owlModuleCreate(context,ptxCode);
  
  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare *sphere* geometry type(s)
  // -------------------------------------------------------

  // ----------- metal -----------
  OWLVarDecl metalSpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(MetalSpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType metalSpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(MetalSpheresGeom),
                        metalSpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(metalSpheresGeomType,0,
                           module,"MetalSpheres");
  owlGeomTypeSetIntersectProg(metalSpheresGeomType,0,
                              module,"MetalSpheres");
  owlGeomTypeSetBoundsProg(metalSpheresGeomType,
                           module,"MetalSpheres");

  // ----------- dielectric -----------
  OWLVarDecl dielectricSpheresGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(DielectricSpheresGeom,prims)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType dielectricSpheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(DielectricSpheresGeom),
                        dielectricSpheresGeomVars,-1);
  owlGeomTypeSetClosestHit(dielectricSpheresGeomType,0,
                           module,"DielectricSpheres");
  owlGeomTypeSetIntersectProg(dielectricSpheresGeomType,0,
                              module,"DielectricSpheres");
  owlGeomTypeSetBoundsProg(dielectricSpheresGeomType,
                           module,"DielectricSpheres");

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



  // -------------------------------------------------------
  // declare *boxes* geometry type(s)
  // -------------------------------------------------------

  // ----------- metal -----------
  OWLVarDecl metalBoxesGeomVars[] = {
    { "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(MetalBoxesGeom,perBoxMaterial)},
    { "vertex",         OWL_BUFPTR, OWL_OFFSETOF(MetalBoxesGeom,vertex)},
    { "index",          OWL_BUFPTR, OWL_OFFSETOF(MetalBoxesGeom,index)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType metalBoxesGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_TRIANGLES,
                        sizeof(MetalBoxesGeom),
                        metalBoxesGeomVars,-1);
  owlGeomTypeSetClosestHit(metalBoxesGeomType,0,
                           module,"MetalBoxes");

  // ----------- dielectric -----------
  OWLVarDecl dielectricBoxesGeomVars[] = {
    { "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(DielectricBoxesGeom,perBoxMaterial)},
    { "vertex",         OWL_BUFPTR, OWL_OFFSETOF(DielectricBoxesGeom,vertex)},
    { "index",          OWL_BUFPTR, OWL_OFFSETOF(DielectricBoxesGeom,index)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType dielectricBoxesGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_TRIANGLES,
                        sizeof(DielectricBoxesGeom),
                        dielectricBoxesGeomVars,-1);
  owlGeomTypeSetClosestHit(dielectricBoxesGeomType,0,
                           module,"DielectricBoxes");

  // ----------- lambertian -----------
  OWLVarDecl lambertianBoxesGeomVars[] = {
    { "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(LambertianBoxesGeom,perBoxMaterial)},
    { "vertex",         OWL_BUFPTR, OWL_OFFSETOF(LambertianBoxesGeom,vertex)},
    { "index",          OWL_BUFPTR, OWL_OFFSETOF(LambertianBoxesGeom,index)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType lambertianBoxesGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_TRIANGLES,
                        sizeof(LambertianBoxesGeom),
                        lambertianBoxesGeomVars,-1);
  owlGeomTypeSetClosestHit(lambertianBoxesGeomType,0,
                           module,"LambertianBoxes");
  

  // -------------------------------------------------------
  // make sure to do that *before* setting up the geometry, since the
  // user geometry group will need the compiled bounds programs upon
  // accelBuild()
  // -------------------------------------------------------
  owlBuildPrograms(context);






  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ====================== SPHERES ======================
  
  // ----------- metal -----------
  OWLBuffer metalSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(metalSpheres[0]),
                            metalSpheres.size(),metalSpheres.data());
  OWLGeom metalSpheresGeom
    = owlGeomCreate(context,metalSpheresGeomType);
  owlGeomSetPrimCount(metalSpheresGeom,metalSpheres.size());
  owlGeomSetBuffer(metalSpheresGeom,"prims",metalSpheresBuffer);

  // ----------- lambertian -----------
  OWLBuffer lambertianSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(lambertianSpheres[0]),
                            lambertianSpheres.size(),lambertianSpheres.data());
  OWLGeom lambertianSpheresGeom
    = owlGeomCreate(context,lambertianSpheresGeomType);
  owlGeomSetPrimCount(lambertianSpheresGeom,lambertianSpheres.size());
  owlGeomSetBuffer(lambertianSpheresGeom,"prims",lambertianSpheresBuffer);

  // ----------- dielectric -----------
  OWLBuffer dielectricSpheresBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(dielectricSpheres[0]),
                            dielectricSpheres.size(),dielectricSpheres.data());
  OWLGeom dielectricSpheresGeom
    = owlGeomCreate(context,dielectricSpheresGeomType);
  owlGeomSetPrimCount(dielectricSpheresGeom,dielectricSpheres.size());
  owlGeomSetBuffer(dielectricSpheresGeom,"prims",dielectricSpheresBuffer);





  // ====================== BOXES ======================
  
  // ----------- metal -----------
  OWLBuffer metalMaterialsBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(metalBoxes.materials[0]),
                            metalBoxes.materials.size(),
                            metalBoxes.materials.data());
  OWLBuffer metalVerticesBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,
                            metalBoxes.vertices.size(),
                            metalBoxes.vertices.data());
  OWLBuffer metalIndicesBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,
                            metalBoxes.indices.size(),
                            metalBoxes.indices.data());
  OWLGeom metalBoxesGeom
    = owlGeomCreate(context,metalBoxesGeomType);
  owlTrianglesSetVertices(metalBoxesGeom,metalVerticesBuffer,
                          metalBoxes.vertices.size(),
                          sizeof(metalBoxes.vertices[0]),0);
  owlTrianglesSetIndices(metalBoxesGeom,metalIndicesBuffer,
                         metalBoxes.indices.size(),
                         sizeof(metalBoxes.indices[0]),0);
  owlGeomSetBuffer(metalBoxesGeom,"perBoxMaterial",metalMaterialsBuffer);
  owlGeomSetBuffer(metalBoxesGeom,"vertex",metalVerticesBuffer);
  owlGeomSetBuffer(metalBoxesGeom,"index",metalIndicesBuffer);
  
  // ----------- lambertian -----------
  OWLBuffer lambertianMaterialsBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(lambertianBoxes.materials[0]),
                            lambertianBoxes.materials.size(),
                            lambertianBoxes.materials.data());
  OWLBuffer lambertianVerticesBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,
                            lambertianBoxes.vertices.size(),
                            lambertianBoxes.vertices.data());
  OWLBuffer lambertianIndicesBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,
                            lambertianBoxes.indices.size(),
                            lambertianBoxes.indices.data());
  OWLGeom lambertianBoxesGeom
    = owlGeomCreate(context,lambertianBoxesGeomType);
  owlTrianglesSetVertices(lambertianBoxesGeom,lambertianVerticesBuffer,
                          lambertianBoxes.vertices.size(),
                          sizeof(lambertianBoxes.vertices[0]),0);
  owlTrianglesSetIndices(lambertianBoxesGeom,lambertianIndicesBuffer,
                         lambertianBoxes.indices.size(),
                         sizeof(lambertianBoxes.indices[0]),0);
  owlGeomSetBuffer(lambertianBoxesGeom,"perBoxMaterial",lambertianMaterialsBuffer);
  owlGeomSetBuffer(lambertianBoxesGeom,"vertex",lambertianVerticesBuffer);
  owlGeomSetBuffer(lambertianBoxesGeom,"index",lambertianIndicesBuffer);
  
  // ----------- dielectric -----------
  OWLBuffer dielectricMaterialsBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(dielectricBoxes.materials[0]),
                            dielectricBoxes.materials.size(),
                            dielectricBoxes.materials.data());
  OWLBuffer dielectricVerticesBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,
                            dielectricBoxes.vertices.size(),
                            dielectricBoxes.vertices.data());
  OWLBuffer dielectricIndicesBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,
                            dielectricBoxes.indices.size(),
                            dielectricBoxes.indices.data());
  OWLGeom dielectricBoxesGeom
    = owlGeomCreate(context,dielectricBoxesGeomType);
  owlTrianglesSetVertices(dielectricBoxesGeom,dielectricVerticesBuffer,
                          dielectricBoxes.vertices.size(),
                          sizeof(dielectricBoxes.vertices[0]),0);
  owlTrianglesSetIndices(dielectricBoxesGeom,dielectricIndicesBuffer,
                         dielectricBoxes.indices.size(),
                         sizeof(dielectricBoxes.indices[0]),0);
  owlGeomSetBuffer(dielectricBoxesGeom,"perBoxMaterial",dielectricMaterialsBuffer);
  owlGeomSetBuffer(dielectricBoxesGeom,"vertex",dielectricVerticesBuffer);
  owlGeomSetBuffer(dielectricBoxesGeom,"index",dielectricIndicesBuffer);
  


  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  // ----------- one group for the spheres -----------
  /* (note these are user geoms, so have to be in another group than the triangle
     meshes) */
  OWLGeom  userGeoms[] = {
    lambertianSpheresGeom,
    metalSpheresGeom,
    dielectricSpheresGeom
  };
  OWLGroup userGeomGroup
    = owlUserGeomGroupCreate(context,3,userGeoms);
  owlGroupBuildAccel(userGeomGroup);

  // ----------- one group for the boxes -----------
  /* (note these are made of triangles, so have to be in another group
     than the sphere geoms) */
  OWLGeom  triangleGeoms[] = {
    lambertianBoxesGeom,
    metalBoxesGeom,
    dielectricBoxesGeom
  };
  OWLGroup triangleGeomGroup
    = owlTrianglesGeomGroupCreate(context,3,triangleGeoms);
  owlGroupBuildAccel(triangleGeomGroup);

  // ----------- one final group with one instance each -----------
  /* (this is just the simplest way of creating triangular with
  non-triangular geometry: create one separate instance each, and
  combine them in a instance group) */
  OWLGroup world =
    owlInstanceGroupCreate(context,2);
  owlInstanceGroupSetChild(world,0,userGeomGroup);
  owlInstanceGroupSetChild(world,1,triangleGeomGroup);
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
  const vec3f origin = lookFrom;
  const vec3f w = normalize(lookFrom - lookAt);
  const vec3f u = normalize(cross(vup, w));
  const vec3f v = cross(w, u);
  const vec3f lower_left_corner
    = origin - half_width * focusDist*u - half_height * focusDist*v - focusDist * w;
  const vec3f horizontal = 2.0f*half_width*focusDist*u;
  const vec3f vertical = 2.0f*half_height*focusDist*v;

  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSetGroup (rayGen,"world",        world);
  owlRayGenSet3f    (rayGen,"camera.org",   (const owl3f&)origin);
  owlRayGenSet3f    (rayGen,"camera.llc",   (const owl3f&)lower_left_corner);
  owlRayGenSet3f    (rayGen,"camera.horiz", (const owl3f&)horizontal);
  owlRayGenSet3f    (rayGen,"camera.vert",  (const owl3f&)vertical);

  owlRayGenSet1i    (rayGen,"deviceCount",  numGPUsFound);
  
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
