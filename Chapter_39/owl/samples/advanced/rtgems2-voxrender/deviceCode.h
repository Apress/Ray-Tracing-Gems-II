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

#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>

using namespace owl;

struct LaunchParams
{
  OptixTraversableHandle world;
  vec3f sunDirection;  // pointing toward sun
  vec3f sunColor;
  float brickScale;

  int clipHeight;  // in units of bricks

  // Note: these are also bound values
  bool enableClipping;
  bool enableToonOutline; 

  int frameID;
  float4   *fbAccumBuffer;
  uint32_t *fbPtr;
  vec2i  fbSize;

};

struct TrianglesGeomData
{
  unsigned char *colorIndexPerBrick;
  uchar4 *colorPalette;

  vec3i *index;
  vec3f *vertex;

  // For flat meshes, so we can get brickId from primId
  int primCountPerBrick;
  
};

// A set of bricks in a 3d grid
// Each brick is stored as indices in the grid plus a color index
struct VoxGeomData {
  uchar4 *prims;  // (xi, yi, zi, ci)
  uchar4 *colorPalette;
  bool enableToonOutline;
};

// A version of VoxGeom where primitives are larger than single bricks.
// Each prim is a "block" of NxNxN bricks
struct VoxBlockGeomData {
  uchar3 *prims;  // xyz indices of LL corners of blocks.  Each block is a cube of bricks.
  unsigned char *colorIndices; 
  uchar4 *colorPalette;

  // No toon outline for this mode
};

/* variables for the ray generation program */
struct RayGenData
{
  struct {
    vec3f pos;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
  } camera;
};


