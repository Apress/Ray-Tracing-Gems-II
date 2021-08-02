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

// This program shows a minimal setup: no geometry, just a ray generation
// shader that accesses the pixels and draws a checkerboard pattern to
// the output file ll00-rayGenOnly.png

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

// When run, this program produces this PNG as output.
// In this case the correct result is a red and light gray checkerboard,
// as nothing is actually rendered
const char *outFileName = "s00-rayGenOnly.png";
// image resolution
const vec2i fbSize(800,600);
// camera: unused in this sample, which generates no rays. TODO: delete?
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

int main(int ac, char **av)
{
  // The output window will show comments for many of the methods called.
  // Walking through the code line by line with a debugger is educational.
  LOG("owl example '" << av[0] << "' starting up");

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");

  // Initialize CUDA and OptiX 7, and create an "owl device," a context to hold the
  // ray generation shader and output buffer. The "1" is the number of devices requested.
  OWLContext owl
    = owlContextCreate(nullptr,1);
  // PTX is the intermediate code that the CUDA deviceCode.cu shader program is converted into.
  // You can see the machine-centric PTX code in
  // build\samples\s00-rayGenOnly\cuda_compile_ptx_1_generated_deviceCode.cu.ptx_embedded.c
  // This PTX intermediate code representation is then compiled into an OptiX module.
  // See https://devblogs.nvidia.com/how-to-get-started-with-optix-7/ for more information.
  OWLModule module
    = owlModuleCreate(owl,ptxCode);

  OWLVarDecl rayGenVars[]
    = {
       { "fbPtr",  OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr) },
       { "fbSize", OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize) },
       { "color0", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,color0) },
       { "color1", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,color1) },
       { /* sentinel: */ nullptr }
  };
  // Allocate room for one RayGen shader, create it, and
  // hold on to it with the "owl" context
  OWLRayGen rayGen
    = owlRayGenCreate(owl,module,"simpleRayGen",
                      sizeof(RayGenData),rayGenVars,-1);
 
  // (re-)builds all optix programs, with current pipeline settings 
  owlBuildPrograms(owl);
  // Create the pipeline. Note that owl will (kindly) warn there are no geometry and no miss programs defined.
  owlBuildPipeline(owl);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  LOG("allocating frame buffer");
  // Create a frame buffer as page-locked, aka "pinned" memory. See CUDA documentation for benefits and more info.
  OWLBuffer
  frameBuffer = owlHostPinnedBufferCreate(owl,
                                          /*type:*/OWL_INT,
                                          /*size:*/fbSize.x*fbSize.y);

  // ------------------------------------------------------------------
  // build Shader Binding Table (SBT) required to trace the groups
  // ------------------------------------------------------------------

  owlRayGenSet3f(rayGen,"color0",.8f,0.f,0.f);
  owlRayGenSet3f(rayGen,"color1",.8f,.8f,.8f);
  owlRayGenSetBuffer(rayGen,"fbPtr",frameBuffer);
  owlRayGenSet2i(rayGen,"fbSize",fbSize.x,fbSize.y);
  // Build a shader binding table entry for the ray generation record.
  owlBuildSBT(owl);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  // Normally launching without a hit or miss shader causes OptiX to trigger warnings.
  // Owl's wrapper call here will set up fake hit and miss records into the SBT to avoid these.
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing frame buffer to " << outFileName);
  const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("cleaning up ...");
  owlModuleRelease(module);
  owlRayGenRelease(rayGen);
  owlBufferRelease(frameBuffer);
  owlContextDestroy(owl);
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
