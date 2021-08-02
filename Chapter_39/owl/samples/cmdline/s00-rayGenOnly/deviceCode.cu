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

// Ray gen shader for ll00-rayGenOnly. No actual rays are harmed in the making of
// this shader. The pixel location is simply translated into a checkerboard pattern.

#include "deviceCode.h"
#include <optix_device.h>

// OPTIX_RAYGEN_PROGRAM() is a simple macro defined in deviceAPI.h to add standard
// code for defining a shader method.
// It puts:
//   extern "C" __global__ void __raygen__##programName
// in front of the program name given
OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  // read in the program data set by the calling program hostCode.cpp using lloSbtRayGensBuild;
  // see RayGenData in deviceCode.h
  const RayGenData &self = owl::getProgramData<RayGenData>();
  // Under the hood, OptiX maps rays generated in CUDA thread blocks to a pixel ID,
  // where the ID is a 2D vector, 0 to frame buffer width-1, 0 to height-1
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    // the first thread ID is always (0,0), so we can generate a message to show things are working
    printf("%sHello OptiX From your First RayGen Program%s\n",
           OWL_TERMINAL_CYAN,
           OWL_TERMINAL_DEFAULT);
  }
  
  // Generate a simple checkerboard pattern as a test. Note that the upper left corner is pixel (0,0).
  int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
  // alternate pattern, showing that pixel (0,0) is in the upper left corner
  // pattern = (pixelID.x*pixelID.x + pixelID.y*pixelID.y) / 100000;
  const vec3f color = (pattern&1) ? self.color1 : self.color0;

  // find the frame buffer location (x + width*y) and put the "computed" result there
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

