#include "Renderer.h"

#include "samples/common/owlViewer/OWLViewer.h"

#include <fstream>
#include <sstream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "./owl/3rdParty/stb_image/stb/stb_image.h"
#include "./owl/3rdParty/stb_image/stb/stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace cdf {
  using owl::viewer::SimpleCamera;

  struct {
    std::string objFileName = "";
    std::string blueNoiseFileName = "";
    std::string outFileName = "owlCDF.png";
    struct {
      vec3f vp = vec3f(0.f);
      vec3f vu = vec3f(0.f);
      vec3f vi = vec3f(0.f);
      float fovy = 80;
    } camera;
    vec2i windowSize = vec2i(800,600);
    std::string benchmarkMode = "none";
    int screenshotFrameNum = 0;
  } cmdline;
  
  void usage(const std::string &err)
  {
    if (err != "")
      std::cout << OWL_TERMINAL_RED << "\nFatal error: " << err
                << OWL_TERMINAL_DEFAULT << std::endl << std::endl;

    std::cout << "Usage: ./owlCDF environmentMap.{exr|hdr} --bn blueNoise_[#.png] --obj object.obj [--rendermode {bvh|binary}] [--simplification-rate (0,1)]" << std::endl;
    std::cout << std::endl;
    exit(1);
  }
  
  struct Viewer : public owl::viewer::OWLViewer {
  public:
    typedef owl::viewer::OWLViewer inherited;
    
    Viewer(Renderer *renderer)
      : inherited("owlCDF Sample Viewer", cmdline.windowSize, /*visible*/true, /*vsync*/false),
        renderer(renderer)
    {
    }
    
    /*! this function gets called whenever the viewer widget changes camera settings */
    void cameraChanged() override;
    void resize(const vec2i &newSize) override;
    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;

    /*! this gets called when the user presses a key on the keyboard ... */
    void key(char key, const vec2i &where) override
    {
      inherited::key(key,where);
      renderer->resetAccum();
      switch (key) {
      case '!':
        std::cout << "saving screenshot to 'owlDVR.png'" << std::endl;
        screenShot("owlDVR.png");
        break;
      case 'H':
        renderer->heatMapEnabled = !renderer->heatMapEnabled;
        break;
      case '<':
        renderer->heatMapScale /= 1.5f;
        break;
      case '>':
        renderer->heatMapScale *= 1.5f;
        break;
      case ')':
        renderer->spp++;
        PRINT(renderer->spp);
        break;
      case '(':
        renderer->spp = max(1,renderer->spp-1);
        PRINT(renderer->spp);
        break;
      case '1':
        renderer->renderMode = RenderModeBVH;
        std::cout<<"Using RTX BVH for ITS"<<std::endl;
        break;
      case '2':
        renderer->renderMode = RenderModeBinarySearch;
        std::cout<<"Using binary search for ITS"<<std::endl;
        break;
      case '3':
        renderer->renderMode = RenderModeRandom;
        std::cout<<"Using random selection"<<std::endl;
        break;
      case '9':
        renderer->hdriIntensity = max(0.f,renderer->hdriIntensity-.1f);
        PRINT(renderer->hdriIntensity);
        break;
      case '0':
        renderer->hdriIntensity = renderer->hdriIntensity+.1f;
        PRINT(renderer->hdriIntensity);
        break;
      }
    }

  public:

    Renderer *const renderer;
  };
  

  void Viewer::resize(const vec2i &newSize) 
  {
    // ... tell parent to resize (also resizes the pbo in the wingdow)
    inherited::resize(newSize);
    cameraChanged();
    renderer->resetAccum();
  }
    
  /*! this function gets called whenever the viewer widget changes
    camera settings */
  void Viewer::cameraChanged() 
  {
    inherited::cameraChanged();
    const SimpleCamera &camera = inherited::getCamera();
    
    const vec3f screen_du = camera.screen.horizontal / float(getWindowSize().x);
    const vec3f screen_dv = camera.screen.vertical   / float(getWindowSize().y);
    const vec3f screen_00 = camera.screen.lower_left;
    renderer->setCamera(camera.lens.center,screen_00,screen_du,screen_dv);
    renderer->resetAccum();
  }
    

  /*! gets called whenever the viewer needs us to re-render out widget */
  void Viewer::render() 
  {
    static double t_last = getCurrentTime();
    static double t_first = t_last;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    renderer->render(fbSize,fbPointer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
      
    double t_now = getCurrentTime();
    static double avg_t = t_now-t_last;
    if (t_last >= 0)
      avg_t = 0.8*avg_t + 0.2*(t_now-t_last);

    size_t numSamples = fbSize.x*fbSize.y*Renderer::spp;
    double numSamplesPerSec = numSamples/(milliseconds / 1000.f);

    static int counter = 0;
    counter++;
    
    if (counter == cmdline.screenshotFrameNum) {
      std::stringstream str;  
      std::string s = cmdline.outFileName;
      screenShot(s.c_str());
      exit(0);
    }

    char title[1000];
    sprintf(title,"RTX ITS - %.2f FPS - SPS %s",(1.f/avg_t), prettyNumber(numSamplesPerSec).c_str());
    setTitle(title);

    t_last = t_now;
  }

  extern "C" int main(int argc, char **argv)
  {
    std::string inFileName;
    
    for (int i=1;i<argc;i++) {
      const std::string arg = argv[i];
      if (arg[0] != '-') {
        inFileName = arg;
      } else if (arg == "--obj" || arg == "-obj") {
        cmdline.objFileName = argv[++i];
      } else if (arg == "--bn" || arg == "-bn") {
        cmdline.blueNoiseFileName = argv[++i];
      } else if (arg == "-fovy") {
        cmdline.camera.fovy = std::stof(argv[++i]);
      }
      else if (arg == "-win") {
        cmdline.windowSize.x = std::stoi(argv[++i]);
        cmdline.windowSize.y = std::stoi(argv[++i]);
      }
      else if (arg == "--camera") {
        cmdline.camera.vp.x = std::stof(argv[++i]);
        cmdline.camera.vp.y = std::stof(argv[++i]);
        cmdline.camera.vp.z = std::stof(argv[++i]);
        cmdline.camera.vi.x = std::stof(argv[++i]);
        cmdline.camera.vi.y = std::stof(argv[++i]);
        cmdline.camera.vi.z = std::stof(argv[++i]);
        cmdline.camera.vu.x = std::stof(argv[++i]);
        cmdline.camera.vu.y = std::stof(argv[++i]);
        cmdline.camera.vu.z = std::stof(argv[++i]);
      }
      else if (arg == "-win"  || arg == "--win" || arg == "--size") {
        cmdline.windowSize.x = std::atoi(argv[++i]);
        cmdline.windowSize.y = std::atoi(argv[++i]);
      }
      else if (arg == "-o") {
        cmdline.outFileName = argv[++i];
      }
      else if (arg == "-spp" || arg == "--spp") {
        Renderer::spp = std::stoi(argv[++i]);
      }
      else if (arg == "--heat-map") {
        Renderer::heatMapEnabled = true;
        Renderer::heatMapScale = std::stof(argv[++i]);
      } else if (arg == "-rm" || arg == "--rendermode") {
        std::string val(argv[++i]);
        if (val == "bvh")
          Renderer::renderMode = RenderModeBVH;
        else if(val == "binary")
          Renderer::renderMode = RenderModeBinarySearch;
        else
          usage("invalid value for argument --rendermode: '"+val+"'");
      } else if (arg == "--benchmark" || arg == "-bench") {
        std::string val(argv[++i]);
        if (val == "bvh")
          Renderer::benchmarkMode = BenchmarkModeBVH;
        else if (val == "binary")
          Renderer::benchmarkMode = BenchmarkModeBinarySearch;
        else if (val == "errors")
          Renderer::benchmarkMode = BenchmarkModeErrors;
        else if (val == "none") // default
          Renderer::benchmarkMode = BenchmarkModeNone;
        else
          usage("invalid value for argument --benchmark: '"+val+"'");
      }  else if (arg == "-dumpObj" || arg == "--dumpObj") {
        Renderer::cdfDumpAsObj = true;
      } else if (arg == "-screenshotfn" || arg == "--screenshotFrameNum") {
        cmdline.screenshotFrameNum = std::stoi(argv[++i]);
      }  else if (arg == "-sr" || arg == "--simplification-rate") {
        Renderer::simplificationRate = std::stof(argv[++i]);
      } else if (arg == "-terminate" || arg == "--terminate") {
        Renderer::terminateRenderer = true;
      }
      else
        usage("unknown cmdline arg '"+arg+"'");
    }
    
    if (inFileName == "")
      usage("no filename specified");

    if (cmdline.blueNoiseFileName == "")
      usage("no blue noise filename specified");

    Renderer renderer(inFileName,cmdline.objFileName,cmdline.blueNoiseFileName);
 
    const box3f modelBounds = renderer.modelBounds;

    Viewer *viewer = new Viewer(&renderer);
    viewer->enableFlyMode();
    viewer->enableInspectMode(/* valid range of poi*/modelBounds,
                              /* min distance      */1e-3f,
                              /* max distance      */1e8f);

    if (cmdline.camera.vu != vec3f(0.f)) {
      viewer->setCameraOrientation(/*origin   */cmdline.camera.vp,
                                   /*lookat   */cmdline.camera.vi,
                                   /*up-vector*/cmdline.camera.vu,
                                   /*fovy(deg)*/cmdline.camera.fovy);
    } else {
      viewer->setCameraOrientation(/*origin   */
                                   modelBounds.center()
                                   + vec3f(-.0f, 1.f, 1.f) * modelBounds.span(),
                                   /*lookat   */modelBounds.center() + vec3f(0.0f, 0.f, .1f) ,
                                   /*up-vector*/vec3f(0.f, 1.f, 0.f),
                                   /*fovy(deg)*/70.f);
    }
    viewer->setWorldScale(10.1f*length(modelBounds.span()));
    
    std::cout<<"press ! to save screenshot"<<std::endl;
    std::cout<<"press H to show heatmap"<<std::endl;
    std::cout<<"press < and > to increase or decrease heatmap scale"<<std::endl;
    std::cout<<"press ( and ) to increase or decrease samples per pixel"<<std::endl;
    std::cout<<"press + and _ to increase or decrease camera speed"<<std::endl;
    std::cout<<"press 9 and 0 to increase or decrease HDRI intensity"<<std::endl;
    std::cout<<"press 1 for RTX accelerated CDF inversion"<<std::endl;
    std::cout<<"press 2 for binary search based CDF inversion"<<std::endl;
    std::cout<<"press 3 for random sampling"<<std::endl;

    viewer->showAndRun();
  }  
}

