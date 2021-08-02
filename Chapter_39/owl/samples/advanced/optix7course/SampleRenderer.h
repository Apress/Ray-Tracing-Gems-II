// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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
#ifdef WIN32
#include <windows.h>
#include <gl/GL.h>
#endif
#include <cuda_gl_interop.h>
// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

#define OWL_TEXTURES 1

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct Camera {
    /*! camera position - *from* where we are looking */
    vec3f from;
    /*! which point we are looking *at* */
    vec3f at;
    /*! general up-vector */
    vec3f up;
  };
  
  /*! a sample OptiX-7 renderer that demonstrates how to set up
      context, module, programs, pipeline, SBT, etc, and perform a
      valid launch that renders some pixel (using a simple test
      pattern, in this case */
  class SampleRenderer
  {
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
  public:
    /*! constructor - performs all setup, including initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    SampleRenderer(const Model *model, const QuadLight &light);

    /*! render one frame */
    void render();

    /*! resize frame buffer to given resolution */
    void resize(void *fbPointer, const vec2i &newSize);

    // /*! download the rendered color buffer */
    // void downloadPixels(uint32_t h_pixels[]);

    /*! set camera to render with */
    void setCamera(const Camera &camera);

    
    bool denoiserOn = true;
    bool accumulate = true;

    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    void computeFinalPixelColors();
    
    /*! helper function that initializes optix and checks for errors */
    void initOptix();
  
    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModule();
    
    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();
    
    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();
    
    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    void buildAccel();

    /*! upload textures, and create cuda texture objects for them */
    void createTextures();

    OWLContext      context = nullptr;
    OWLModule       module = nullptr;
    OWLLaunchParams launchParams = nullptr;
    OWLRayGen       rayGen = nullptr;
    OWLMissProg     missProgRadiance = nullptr;
    OWLMissProg     missProgShadow = nullptr;

    /*! the color buffer we use during _rendering_, which is a bit
      larger than the actual displayed frame buffer (to account for
      the border), and in float4 format (the denoiser requires
      floats) */
    OWLBuffer fbColor = nullptr;
    OWLBuffer fbNormal = nullptr;
    OWLBuffer fbAlbedo = nullptr;
    
    /*! output of the denoiser pass, in float4 */
    OWLBuffer denoisedBuffer = nullptr;
    
    /* the actual final color buffer used for display, in rgba8 */
    // OWLBuffer fbFinal = nullptr;

    OWLBuffer    denoiserScratch = nullptr;
    OWLBuffer    denoiserState = nullptr;
    OWLBuffer    denoiserIntensity = nullptr;
    
    /*! the camera we are to render with. */
    Camera lastSetCamera;
    int    frameID = 0;
    int    numPixelSamples = 1;
    vec2i  fbSize;
    
    /*! the model we are going to trace rays against */
    const Model *model;
    
    /*! @{ one buffer per input mesh */
    // std::vector<OWLBuffer> vertexBuffer;
    // std::vector<OWLBuffer> normalBuffer;
    // std::vector<OWLBuffer> texcoordBuffer;
    // std::vector<OWLBuffer> indexBuffer;
    /*! @} */
    
    OWLGroup world = nullptr;

    /*! @{ one texture object and pixel array per used texture */
#if OWL_TEXTURES
    std::vector<OWLTexture> textures;
#else
    std::vector<cudaArray_t>         textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
#endif
    /*! @} */
  };

} // ::osc
