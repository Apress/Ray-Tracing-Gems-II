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

#include "../../cmdline/s05-rtow/Materials.h"

namespace owl {
  
  // ==================================================================
  /* the raw geometric shape of a sphere, without material - this is
     what goes into intersection and bounds programs */
  // ==================================================================
  struct Sphere {
    vec3f center;
    float radius;
  };

  // ==================================================================
  /* the three actual primitive types created by fusing material data
     and geometry data */
  // ==================================================================

  struct MetalSphere {
    Sphere sphere;
    Metal  material;
  };
  struct DielectricSphere {
    Sphere sphere;
    Dielectric material;
  };
  struct LambertianSphere {
    Sphere sphere;
    Lambertian material;
  };

  // ==================================================================
  /* the three actual "Geoms" that each consist of multiple prims of
     same type (this is what optix6 would have called the "geometry
     instance" */
  // ==================================================================

  struct MetalSpheresGeom {
    /* for spheres geometry we store one full "sphere+material" record
       per sphere */
    MetalSphere *prims;
  };
  struct DielectricSpheresGeom {
    /* for spheres geometry we store one full "sphere+material" record
       per sphere */
    DielectricSphere *prims;
  };
  struct LambertianSpheresGeom {
    /* for spheres geometry we store one full "sphere+material" record
       per sphere */
    LambertianSphere *prims;
  };

  struct MetalBoxesGeom {
    /*! for our boxes geometry we use triangles for the geometry, so the
      materials will actually be shared among every group of 12
      triangles */
    Metal *perBoxMaterial;
    /* the vertex and index arrays for the triangle mesh */
    vec3f *vertex;
    vec3i *index;
  };
  struct DielectricBoxesGeom {
    /*! for our boxes geometry we use triangles for the geometry, so the
      materials will actually be shared among every group of 12
      triangles */
    Dielectric *perBoxMaterial;
    /* the vertex and index arrays for the triangle mesh */
    vec3f *vertex;
    vec3i *index;
  };
  struct LambertianBoxesGeom {
    /*! for our boxes geometry we use triangles for the geometry, so the
      materials will actually be shared among every group of 12
      triangles */
    Lambertian *perBoxMaterial;
    /* the vertex and index arrays for the triangle mesh */
    vec3f *vertex;
    vec3i *index;
  };

  // ==================================================================
  /* and finally, input for raygen and miss programs */
  // ==================================================================
  struct RayGenData
  {
    uint32_t *fbPtr;
    float4   *accumBuffer;
    vec2i  fbSize;
    OptixTraversableHandle world;
    int    accumID;
    struct {
      vec3f origin;
      vec3f lower_left_corner;
      vec3f horizontal;
      vec3f vertical;
    } camera;
  };

  struct MissProgData
  {
    /* nothing in this example */
  };

}
