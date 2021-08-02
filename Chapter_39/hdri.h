#pragma once

#include <vector>
#include <cstring>
#include <stdexcept>

#include "samples/common/3rdParty/stb/stb_image.h"
#include "samples/common/3rdParty/stb/stb_image_write.h"
#include "tinyexr.h"

namespace cdf {
  struct HDRI
  {
    inline float const *samplePoint(int x, int y) const
    {
      return pixel.data() + (y * width + x) * numComponents;
    }

    void load(std::string fileName) {
      // check the extension
      std::string extension = std::string(strrchr(fileName.c_str(), '.'));
      std::transform(extension.data(), extension.data() + extension.size(), 
          std::addressof(extension[0]), [](unsigned char c){ return std::tolower(c); });

      if (extension.compare(".hdr") != 0 && extension.compare(".exr") != 0) {
        throw std::runtime_error("Error: expected either a .hdr or a .exr file");
      }

      if (extension.compare(".hdr") == 0) {
        int w, h, n;
        float *imgData = stbi_loadf(fileName.c_str(), &w, &h, &n, STBI_rgb);

        if (!imgData) throw std::runtime_error("Error: unable to load .hdr");
        width = w;
        height = h;
        numComponents = n;
        pixel.resize(w*h*n);
        memcpy(pixel.data(),imgData,w*h*n*sizeof(float));
        stbi_image_free(imgData);
      } else {
        int w, h, n;
        float* imgData;
        const char* err;
        int ret = LoadEXR(&imgData, &w, &h, fileName.c_str(), &err);
        if (ret != 0)
          throw std::runtime_error(std::string("Error, ") + std::string(err));
        n = 4;

        width = w;
        height = h;
        numComponents = n;
        pixel.resize(w*h*n);
        memcpy(pixel.data(),imgData,w*h*n*sizeof(float));
      }
    }
    void save(std::string fileName) {
      std::string extension = std::string(strrchr(fileName.c_str(), '.'));
      std::transform(extension.data(), extension.data() + extension.size(), 
          std::addressof(extension[0]), [](unsigned char c){ return std::tolower(c); });

      if (extension != ".hdr")
        throw std::runtime_error(std::string("Unsupported extension ")+extension);

      stbi_write_hdr(fileName.c_str(),width,height,numComponents,pixel.data());
    }

    unsigned width;
    unsigned height;
    unsigned numComponents;
    std::vector<float> pixel;
  };
}
