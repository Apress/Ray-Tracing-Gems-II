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

// The Ray Tracing in One Weekend scene, but with cubes substituted for some
// spheres. This program shows how different geometric types in a single scene
// are handled.

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "GeomTypes.h"
// viewer base class, for window and user interaction
#include "owlViewer/OWLViewer.h"

#include <random>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

const vec2i init_fbSize(1600, 800);
const vec3f init_lookFrom(-4.3f, 3.3f, -5.0f);
const vec3f init_lookAt(15.0f, 2.f, -5.0f);
const vec3f init_lookUp(0.0f, 1.0f, 0.0f);
const float init_fovy = 60.f;

std::vector<DielectricSphere> dielectricSpheres;
std::vector<LambertianSphere> lambertianSpheres;
std::vector<MetalSphere>      metalSpheres;


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

inline vec3f rnd3f() { return vec3f(rnd(), rnd(), rnd()); }



template<typename BoxArray, typename Material>
void addBox(BoxArray& boxes,
    const vec3f& center,
    const vec3f size,
    const vec3f axis,
    const float angle,
    const Material& material)
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


    owl::affine3f xfm;

    xfm = owl::affine3f(owl::linear3f::scale(size)) * xfm;
    xfm = owl::affine3f(owl::linear3f::rotate(axis, angle)) * xfm;

    xfm = owl::affine3f(owl::affine3f::translate(center)) * xfm;

    const int startIndex = (int)boxes.vertices.size();
    for (int i = 0; i < NUM_VERTICES; i++)
        boxes.vertices.push_back(owl::xfmPoint(xfm, unitBoxVertices[i]));
    for (int i = 0; i < NUM_INDICES; i++)
        boxes.indices.push_back(unitBoxIndices[i] + vec3i(startIndex));
    boxes.materials.push_back(material);
}

void createScene()
{


    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            bool odd =  ((i + j) % 2) == 1;
            vec3f color = (odd) ? vec3f(1.f, 0.f, 0.f) : vec3f(1.f, 1.f, 0.f);

            float x = j * 1.4;
            float z = i - 16 ;
            addBox(lambertianBoxes, { x, 0.f, z }, { 0.7f, 0.f, 0.5f }, { 1.f, 0.f, 0.f }, { 0.f }, Lambertian{ color });
        }
    }



    dielectricSpheres.push_back({ Sphere{vec3f(2.8f, 3.3f, -4.9f), 1.7f},
          Dielectric{1.5f} });


    metalSpheres.push_back({ Sphere{vec3f(5.5f, 2.f, -7.7f), 1.6f},
          Metal{vec3f(1.f, 1.f, 1.f), 0.1f} });
}



struct Viewer : public owl::viewer::OWLViewer
{
    Viewer();

    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;

    /*! window notifies us that we got resized. We HAVE to override
        this to know our actual render dimensions, and get pointer
        to the device frame buffer that the viewer cated for us */
    void resize(const vec2i& newSize) override;

    /*! this function gets called whenever any camera manipulator
      updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;

    OWLRayGen  rayGen{ 0 };
    OWLContext context{ 0 };
    OWLGroup   world{ 0 };
    OWLBuffer  accumBuffer{ 0 };
    int        accumID{ 0 };
};


/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt = camera.getAt();
    const vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();
    const float vfov = owl::viewer::toDegrees(acosf(cosFovy));
    // ........... compute variable values  ..................
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
        = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
    const vec3f horizontal = 2.0f * half_width * focusDist * u;
    const vec3f vertical = 2.0f * half_height * focusDist * v;

    accumID = 0;

    // ----------- set variables  ----------------------------
    owlRayGenSetGroup(rayGen, "world", world);
    owlRayGenSet3f(rayGen, "camera.org", (const owl3f&)origin);
    owlRayGenSet3f(rayGen, "camera.llc", (const owl3f&)lower_left_corner);
    owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f&)horizontal);
    owlRayGenSet3f(rayGen, "camera.vert", (const owl3f&)vertical);
}

void Viewer::render()
{
    owlRayGenSet1i(rayGen, "accumID", accumID);
    accumID++;
    owlBuildSBT(context);
    owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
}


/*! window notifies us that we got resized */
void Viewer::resize(const vec2i& newSize)
{
    OWLViewer::resize(newSize);
    cameraChanged();

    if (accumBuffer)
        owlBufferResize(accumBuffer, newSize.x * newSize.y * sizeof(float4));
    else
        accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4,
            newSize.x * newSize.y, nullptr);

    owlRayGenSetBuffer(rayGen, "accumBuffer", accumBuffer);
    owlRayGenSet1ul(rayGen, "fbPtr", (uint64_t)fbPointer);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i&)fbSize);

}

Viewer::Viewer()
    : OWLViewer("OWL Whitted Sample",
        init_fbSize)
{
    // ##################################################################
    // init owl
    // ##################################################################

    context = owlContextCreate(nullptr, 1);
    OWLModule  module = owlModuleCreate(context, deviceCode_ptx);

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
            metalSpheresGeomVars, -1);
    owlGeomTypeSetClosestHit(metalSpheresGeomType, 0,
        module, "MetalSpheres");
    owlGeomTypeSetIntersectProg(metalSpheresGeomType, 0,
        module, "MetalSpheres");
    owlGeomTypeSetBoundsProg(metalSpheresGeomType,
        module, "MetalSpheres");

    // ----------- dielectric -----------
    OWLVarDecl dielectricSpheresGeomVars[] = {
      { "prims",  OWL_BUFPTR, OWL_OFFSETOF(DielectricSpheresGeom,prims)},
      { /* sentinel to mark end of list */ }
    };
    OWLGeomType dielectricSpheresGeomType
        = owlGeomTypeCreate(context,
            OWL_GEOMETRY_USER,
            sizeof(DielectricSpheresGeom),
            dielectricSpheresGeomVars, -1);
    owlGeomTypeSetClosestHit(dielectricSpheresGeomType, 0,
        module, "DielectricSpheres");
    owlGeomTypeSetIntersectProg(dielectricSpheresGeomType, 0,
        module, "DielectricSpheres");
    owlGeomTypeSetBoundsProg(dielectricSpheresGeomType,
        module, "DielectricSpheres");





    // -------------------------------------------------------
    // declare *boxes* geometry type(s)
    // -------------------------------------------------------





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
            lambertianBoxesGeomVars, -1);
    owlGeomTypeSetClosestHit(lambertianBoxesGeomType, 0,
        module, "LambertianBoxes");


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
        = owlDeviceBufferCreate(context, OWL_USER_TYPE(metalSpheres[0]),
            metalSpheres.size(), metalSpheres.data());
    OWLGeom metalSpheresGeom
        = owlGeomCreate(context, metalSpheresGeomType);
    owlGeomSetPrimCount(metalSpheresGeom, metalSpheres.size());
    owlGeomSetBuffer(metalSpheresGeom, "prims", metalSpheresBuffer);


    // ----------- dielectric -----------
    OWLBuffer dielectricSpheresBuffer
        = owlDeviceBufferCreate(context, OWL_USER_TYPE(dielectricSpheres[0]),
            dielectricSpheres.size(), dielectricSpheres.data());
    OWLGeom dielectricSpheresGeom
        = owlGeomCreate(context, dielectricSpheresGeomType);
    owlGeomSetPrimCount(dielectricSpheresGeom, dielectricSpheres.size());
    owlGeomSetBuffer(dielectricSpheresGeom, "prims", dielectricSpheresBuffer);


    // ====================== BOXES ======================


    // ----------- lambertian -----------
    OWLBuffer lambertianMaterialsBuffer
        = owlDeviceBufferCreate(context, OWL_USER_TYPE(lambertianBoxes.materials[0]),
            lambertianBoxes.materials.size(),
            lambertianBoxes.materials.data());
    OWLBuffer lambertianVerticesBuffer
        = owlDeviceBufferCreate(context, OWL_FLOAT3,
            lambertianBoxes.vertices.size(),
            lambertianBoxes.vertices.data());
    OWLBuffer lambertianIndicesBuffer
        = owlDeviceBufferCreate(context, OWL_INT3,
            lambertianBoxes.indices.size(),
            lambertianBoxes.indices.data());
    OWLGeom lambertianBoxesGeom
        = owlGeomCreate(context, lambertianBoxesGeomType);
    owlTrianglesSetVertices(lambertianBoxesGeom, lambertianVerticesBuffer,
        lambertianBoxes.vertices.size(),
        sizeof(lambertianBoxes.vertices[0]), 0);
    owlTrianglesSetIndices(lambertianBoxesGeom, lambertianIndicesBuffer,
        lambertianBoxes.indices.size(),
        sizeof(lambertianBoxes.indices[0]), 0);
    owlGeomSetBuffer(lambertianBoxesGeom, "perBoxMaterial", lambertianMaterialsBuffer);
    owlGeomSetBuffer(lambertianBoxesGeom, "vertex", lambertianVerticesBuffer);
    owlGeomSetBuffer(lambertianBoxesGeom, "index", lambertianIndicesBuffer);



    // ##################################################################
    // set up all *ACCELS* we need to trace into those groups
    // ##################################################################

    // ----------- one group for the spheres -----------
    /* (note these are user geoms, so have to be in another group than the triangle
       meshes) */
    OWLGeom  userGeoms[] = {
      metalSpheresGeom,
       dielectricSpheresGeom

    };
    OWLGroup userGeomGroup
        = owlUserGeomGroupCreate(context, 2, userGeoms);
    owlGroupBuildAccel(userGeomGroup);

    // ----------- one group for the boxes -----------
    /* (note these are made of triangles, so have to be in another group
       than the sphere geoms) */
    OWLGeom  triangleGeoms[] = {
      lambertianBoxesGeom,

    };
    OWLGroup triangleGeomGroup
        = owlTrianglesGeomGroupCreate(context, 1, triangleGeoms);
    owlGroupBuildAccel(triangleGeomGroup);

    // ----------- one final group with one instance each -----------
    /* (this is just the simplest way of creating triangular with
    non-triangular geometry: create one separate instance each, and
    combine them in a instance group) */
    world =
        owlInstanceGroupCreate(context, 2);
    owlInstanceGroupSetChild(world, 0, userGeomGroup);
    owlInstanceGroupSetChild(world, 1, triangleGeomGroup);
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
        = owlMissProgCreate(context, module, "miss", sizeof(MissProgData),
            missProgVars, -1);
    owlMissProgSet(context, 0, missProg);

    // ........... set variables  ............................
    /* nothing to set */

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl rayGenVars[] = {
      { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
      { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(RayGenData,accumBuffer)},
      { "accumID",       OWL_INT,    OWL_OFFSETOF(RayGenData,accumID)},
      { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
      { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
      { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.origin)},
      { "camera.llc",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.lower_left_corner)},
      { "camera.horiz",  OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.horizontal)},
      { "camera.vert",   OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.vertical)},
      { /* sentinel to mark end of list */ }
    };

    // ........... create object  ............................
    rayGen
        = owlRayGenCreate(context, module, "rayGen",
            sizeof(RayGenData),
            rayGenVars, -1);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################

    // programs have been built before, but have to rebuild raygen and
    // miss progs
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

int main(int ac, char** av)
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

    Viewer viewer;
    viewer.camera.setOrientation(init_lookFrom,
        init_lookAt,
        init_lookUp,
        init_fovy);
    viewer.enableFlyMode();
    viewer.enableInspectMode(/* the big sphere in the middle: */
        owl::box3f(vec3f(-1, 0, -1), vec3f(1, 2, 1)));
    viewer.showAndRun();

    LOG("destroying devicegroup ...");
    owlContextDestroy(viewer.context);

    LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
