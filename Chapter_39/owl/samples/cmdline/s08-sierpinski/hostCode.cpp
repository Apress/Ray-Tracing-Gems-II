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

// This program renders the recursive Sierpinski tetrahedron to a given depth.
// The code demonstrates how to create nested instances.

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <vector>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

std::vector<vec3f> vertices =
  {
    { -0.5f,-0.5f,-0.5f },
    { +0.5f,-0.5f,-0.5f },
    { +0.5f,+0.5f,-0.5f },
    { -0.5f,+0.5f,-0.5f },
    { 0.0f,0.0f,+0.5f },
  };

std::vector<vec3i> indices =
  {
    { 0,1,3 }, { 1,2,3 },
    { 0,4,1 }, { 0,3,4 },
    { 3,2,4 }, { 1,4,2 },
  };

const char *outFileName = "s08-sierpinski.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(2.f,1.3f,.8f);
const vec3f lookAt(0.f,0.f,-.2f);
const vec3f lookUp(0.f,0.f,1.f);
const float fovy = 30.f;

int main(int ac, char **av)
{
  /* iw, feb 29 - after update to latest driver this sample no longer
     works for numLevels > 2. Currently investigating if this is owl
     issue, or a driver issue */
  uint32_t numLevels = 6;
  LOG("owl example '" << av[0] << "' starting up");

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "--num-levels" || arg == "-nl")
      numLevels = std::atoi(av[++i]);
    else
      throw std::runtime_error("unknown cmdline argument '"+arg+"'");
  }
  if (numLevels < 1)
    throw std::runtime_error("num levels must be 1 or greater");


  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  OWLContext owl
    = owlContextCreate(nullptr,1);
  owlSetMaxInstancingDepth(owl,numLevels);

  OWLModule module
    = owlModuleCreate(owl,deviceCode_ptx);

  // ------------------------------------------------------------------
  OWLVarDecl lambertianMeshTypeVars[]
    = {
       { "material", OWL_BUFPTR, OWL_OFFSETOF(LambertianTriangleMesh,material) },
       { "index",    OWL_BUFPTR, OWL_OFFSETOF(LambertianTriangleMesh,index) },
       { "vertex",   OWL_BUFPTR, OWL_OFFSETOF(LambertianTriangleMesh,vertex) },
       { /* sentinel: */ nullptr }
  };
  OWLGeomType lambertianMeshType
    = owlGeomTypeCreate(owl,OWL_GEOMETRY_TRIANGLES,
                        sizeof(LambertianTriangleMesh),
                        lambertianMeshTypeVars,-1);
  owlGeomTypeSetClosestHit(lambertianMeshType,0,
                           module,"PyramidMesh");

  // ------------------------------------------------------------------
  OWLVarDecl rayGenVars[]
    = {
       { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr) },
       { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize) },
       { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world) },
       { "camera.origin", OWL_FLOAT3,
         OWL_OFFSETOF(RayGenData,camera.origin) },
       { "camera.lower_left_corner", OWL_FLOAT3,
         OWL_OFFSETOF(RayGenData,camera.lower_left_corner) },
       { "camera.horizontal", OWL_FLOAT3,
         OWL_OFFSETOF(RayGenData,camera.horizontal) },
       { "camera.vertical", OWL_FLOAT3,
         OWL_OFFSETOF(RayGenData,camera.vertical) },
       { /* sentinel: */ nullptr }
  };
  OWLRayGen rayGen
    = owlRayGenCreate(owl,module,"renderFrame",
                      sizeof(RayGenData),rayGenVars,-1);

  // ------------------------------------------------------------------
  OWLVarDecl missVars[]
    = {
       /* nothing, in this example */
       { /* sentinel: */ nullptr }
  };
  OWLMissProg miss
    = owlMissProgCreate(owl,module,"miss",
                        sizeof(MissProgData),missVars,-1);
  owlMissProgSet(owl,0,miss);


  // ------------------------------------------------------------------
  owlBuildPrograms(owl);
  owlBuildPipeline(owl);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################
  std::vector<Lambertian> materials;
  Lambertian green;
  green.albedo = owl::vec3f(0,.7f,0);
  materials.push_back(green);

  // ------------------------------------------------------------------
  // create buffers
  // ------------------------------------------------------------------
  OWLBuffer materialsBuffer
    = owlDeviceBufferCreate(owl,
                            OWL_USER_TYPE(Lambertian),
                            materials.size(),
                            materials.data());
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(owl,OWL_FLOAT3,
                            vertices.size(),vertices.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(owl,OWL_INT3,
                            indices.size(),indices.data());
  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(owl,OWL_UINT,fbSize.x*fbSize.y);

  // ------------------------------------------------------------------
  // create actual geometry
  // ------------------------------------------------------------------
  OWLGeom geom
    = owlGeomCreate(owl,lambertianMeshType);
  owlTrianglesSetVertices(geom,vertexBuffer,vertices.size(),sizeof(vertices[0]),0);
  owlTrianglesSetIndices(geom,indexBuffer,indices.size(),sizeof(indices[0]),0);
  owlGeomSetBuffer(geom,"material",materialsBuffer);
  owlGeomSetBuffer(geom,"vertex",vertexBuffer);
  owlGeomSetBuffer(geom,"index",indexBuffer);

  // ------------------------------------------------------------------
  // set up all accel(s) we need to trace into those groups
  // ------------------------------------------------------------------
  OWLGroup meshGroup
    = owlTrianglesGeomGroupCreate(owl,1,&geom);
  owlGroupBuildAccel(meshGroup);

  OWLGroup world = meshGroup;
  for (int level=1;level<(int)numLevels;level++) {
    OWLGroup childGroup = world;
    owl::affine3f xfms[5]
      = {
         owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
         * owl::affine3f::translate(owl::vec3f(-.5f, -.5f, -.5f)),
         owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
         * owl::affine3f::translate(owl::vec3f(+.5f, -.5f, -.5f)),
         owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
         * owl::affine3f::translate(owl::vec3f(-.5f, +.5f, -.5f)),
         owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
         * owl::affine3f::translate(owl::vec3f(+.5f, +.5f, -.5f)),
         owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
         * owl::affine3f::translate(owl::vec3f(0.0f, 0.0, +.5f))
    };
    OWLGroup group
      = owlInstanceGroupCreate(owl,5);
    for (int i=0;i<5;i++) {
      owlInstanceGroupSetChild(group,i,childGroup);
      owlInstanceGroupSetTransform(group,i,
                                   (const float *)&xfms[i],
                                   OWL_MATRIX_FORMAT_OWL);
    }
    owlGroupBuildAccel(group);
    world = group;
  }

  owlRayGenSetGroup(rayGen,"world",world);
  owlRayGenSetBuffer(rayGen,"fbPtr",frameBuffer);
  owlRayGenSet2i(rayGen,"fbSize",fbSize.x,fbSize.y);

  // compute camera frame:
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

  owlRayGenSet3f(rayGen,"camera.origin",(const owl3f&)origin);
  owlRayGenSet3f(rayGen,"camera.lower_left_corner", (const owl3f&)lower_left_corner);
  owlRayGenSet3f(rayGen,"camera.horizontal", (const owl3f&)horizontal);
  owlRayGenSet3f(rayGen,"camera.vertical", (const owl3f&)vertical);

  // ------------------------------------------------------------------
  // build shader binding table required to trace the groups
  // ------------------------------------------------------------------
  LOG("building SBT ...");
  owlBuildSBT(owl);
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
  LOG("done with launch, writing picture ...");

  // for host pinned mem it doesn't matter which device we query:
  const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################

  LOG("destroying devicegroup ...");
  owlContextDestroy(owl);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
