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

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif

#define BLOCKLEN 8  // in bricks, per side
#define OUTLINE_SCALE 1.2f

enum VisibilityMasks {
  VISIBILITY_RADIANCE = 1u << 0,
  VISIBILITY_SHADOW   = 1u << 1,
  VISIBILITY_OUTLINE  = 1u << 2,
  VISIBILITY_ALL = 255
};

// NOTE: the brick geometry here must lie in a unit bounding box in [0,1]x[0,1]x[0,1]
// and have winding order so that normals point outward

#if 1
constexpr int NUM_BRICK_VERTICES = 8;
vec3f brickVertices[NUM_BRICK_VERTICES] =
  {
    { 0.f, 0.f, 0.f },
    { 1.f, 0.f, 0.f },
    { 0.f, 1.f, 0.f },
    { 1.f, 1.f, 0.f },
    { 0.f, 0.f, 1.f },
    { 1.f, 0.f, 1.f },
    { 0.f, 1.f, 1.f },
    { 1.f, 1.f, 1.f }
  };

constexpr int NUM_BRICK_FACES = 12;
vec3i brickIndices[NUM_BRICK_FACES] =
  {
    { 3,1,0 }, { 2,3,0 },
    { 5,7,6 }, { 6,4,5 },
    { 5,4,0 }, { 0,1,5 },
    { 7,3,2 }, { 2,6,7 },
    { 7,5,1 }, { 1,3,7 },
    { 2,0,4 }, { 4,6,2 }
  };

#else
  // Beveled brick created in Blender
  
  constexpr int NUM_BRICK_VERTICES = 24;
  vec3f brickVertices[NUM_BRICK_VERTICES] =
  {
    { 0.950000f, 1.000000f, 0.950000f },
    { 0.950000f, 0.950000f, 1.000000f },
    { 1.000000f, 0.950000f, 0.950000f },
    { 0.950000f, 0.950000f, 0.000000f },
    { 0.950000f, 1.000000f, 0.050000f },
    { 1.000000f, 0.950000f, 0.050000f },
    { 1.000000f, 0.050000f, 0.950000f },
    { 0.950000f, 0.050000f, 1.000000f },
    { 0.950000f, 0.000000f, 0.950000f },
    { 1.000000f, 0.050000f, 0.050000f },
    { 0.950000f, 0.000000f, 0.050000f },
    { 0.950000f, 0.050000f, 0.000000f },
    { 0.050000f, 1.000000f, 0.950000f },
    { 0.000000f, 0.950000f, 0.950000f },
    { 0.050000f, 0.950000f, 1.000000f },
    { 0.000000f, 0.950000f, 0.050000f },
    { 0.050000f, 1.000000f, 0.050000f },
    { 0.050000f, 0.950000f, 0.000000f },
    { 0.000000f, 0.050000f, 0.950000f },
    { 0.050000f, 0.000000f, 0.950000f },
    { 0.050000f, 0.050000f, 1.000000f },
    { 0.050000f, 0.050000f, 0.000000f },
    { 0.050000f, 0.000000f, 0.050000f },
    { 0.000000f, 0.050000f, 0.050000f }
  };

constexpr int NUM_BRICK_FACES = 44;
vec3i brickIndices[NUM_BRICK_FACES] =
  {
    {3, 21, 17 },
    {18, 15, 23 },
    {8, 22, 10 },
    {14, 7, 1 },
    {2, 9, 5 },
    {0, 1, 2 },
    {3, 4, 5 },
    {6, 7, 8 },
    {9, 10, 11 },
    {12, 13, 14 },
    {15, 16, 17 },
    {18, 19, 20 },
    {21, 22, 23 },
    {21, 15, 17 },
    {17, 4, 3 },
    {5, 0, 2 },
    {19, 23, 22 },
    {10, 6, 8 },
    {16, 13, 12 },
    {20, 8, 7 },
    {7, 2, 1 },
    {11, 22, 21 },
    {14, 18, 20 },
    {1, 12, 14 },
    {3, 9, 11 },
    {12, 4, 16 },
    {3, 11, 21 },
    {18, 13, 15 },
    {8, 19, 22 },
    {14, 20, 7 },
    {2, 6, 9 },
    {21, 23, 15 },
    {17, 16, 4 },
    {5, 4, 0 },
    {19, 18, 23 },
    {10, 9, 6 },
    {16, 15, 13 },
    {20, 19, 8 },
    {7, 6, 2 },
    {11, 10, 22 },
    {14, 13, 18 },
    {1, 0, 12 },
    {3, 5, 9 },
    {12, 0, 4 }
  };
#endif


