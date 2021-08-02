// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#include "SampleRenderer.h"

// our helper library for window handling
#include "owlViewer/OWLViewer.h"
#include <GL/gl.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct SampleWindow : public owl::viewer::OWLViewer
  {
    SampleWindow(const std::string &title,
                 const Model *model,
                 const Camera &camera,
                 const QuadLight &light,
                 const float worldScale)
      : OWLViewer(title// ,camera.from,camera.at,camera.up,worldScale
                  ),
        sample(model,light)
    {
      this->camera.setOrientation(camera.from,
                                  camera.at,
                                  camera.up,
                                  60.f);
      this->setWorldScale(worldScale);
      sample.setCamera(camera);
    }
    
    virtual void render() override
    {
      if (camera.lastModified != 0) {
        
        sample.setCamera(Camera{ camera.getFrom(),
                                 camera.getAt(),
                                 camera.getUp() });
        camera.lastModified = 0;
      }
      sample.render();
    }
    
    void resize(const vec2i &newSize) override
    {
      OWLViewer::resize(newSize);
      // fbSize = newSize;
      sample.resize(fbPointer,newSize);
    }

    void key(char key, const vec2i &where) override
    {
      if (key == 'D' || key == ' ') {
        sample.denoiserOn = !sample.denoiserOn;
        std::cout << "denoising now " << (sample.denoiserOn?"ON":"OFF") << std::endl;
        return;
      }
      if (key == 'A') {
        sample.accumulate = !sample.accumulate;
        std::cout << "accumulation/progressive refinement now " << (sample.accumulate?"ON":"OFF") << std::endl;
        return;
      }
      if (key == ',') {
        sample.numPixelSamples
          = std::max(1,sample.numPixelSamples-1);
        std::cout << "num samples/pixel now "
                  << sample.numPixelSamples << std::endl;
        return;
      }
      if (key == '.') {
        sample.numPixelSamples
          = std::max(1,sample.numPixelSamples+1);
        std::cout << "num samples/pixel now "
                  << sample.numPixelSamples << std::endl;
        return;
      }
      OWLViewer::key(key,where);
    }
    
    // vec2i                 fbSize;
    // GLuint                fbTexture {0};
    // cudaGraphicsResource_t cuDisplayTexture;
    SampleRenderer        sample;
    // std::vector<uint32_t> pixels;
  };
  
  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    std::string inFileName = 
#ifdef _WIN32
      // on windows, visual studio creates _two_ levels of build dir
      // (x86/Release)
      "../../models/sponza.obj"
#else
      // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "./sponza.obj"
      //      "../models/sponza.obj"
#endif
      ;
    if (ac == 2)
      inFileName = av[1];
    try {
      Model *model = loadOBJ(inFileName);
      Camera camera = { /*from*/vec3f(-1293.07f, 154.681f, -0.7304f),
                        /* at */model->bounds.center()-vec3f(0,400,0),
                        /* up */vec3f(0.f,1.f,0.f) };

      // some simple, hard-coded light ... obviously, only works for sponza
      const float light_size = 200.f;
      QuadLight light = { /* origin */ vec3f(-1000-light_size,800,-light_size),
                          /* edge 1 */ vec3f(2.f*light_size,0,0),
                          /* edge 2 */ vec3f(0,0,2.f*light_size),
                          /* power */  vec3f(3000000.f) };
                      
      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = length(model->bounds.span());

      SampleWindow *window = new SampleWindow("Optix 7 Course Example (on OWL)",
                                              model,camera,light,worldScale);
      window->enableFlyMode();
      
      std::cout << "Press 'A' to enable/disable accumulation/progressive refinement" << std::endl;
      std::cout << "Press 'D' to enable/disable denoising" << std::endl;
      std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
      std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
      window->showAndRun();
      
    } catch (std::runtime_error& e) {
      std::cout << OWL_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << OWL_TERMINAL_DEFAULT << std::endl;
	  std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
	  exit(1);
    }
    return 0;
  }
  
} // ::osc
