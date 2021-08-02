#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include <vector>
#include <map>
#include <stdexcept>
#include <string>
#include <iostream>
#include <memory>
#include "hdri.h"
#include "mesh.h"

namespace cdf {
  using namespace owl;
  using namespace owl::common;

  enum PixelFormat { PF_RGB32F, PF_RGBA32F };

  template <typename InIt, typename OutIt>
  void scan(InIt first, InIt last, OutIt dest, unsigned stride = 1)
  {
    for (ptrdiff_t i = 0; i != last-first; i += stride)
    {
      *(dest + i) = i == 0 ? *first : *(dest + i - stride) + *(first + i);
    }
  }

  template <typename It>
  void normalize(It first, It last)
  {
    // Assumes that [first,last) is sorted!
    auto bck = *(last-1);
    if (bck != 0)
    {
      for (It it = first; it != last; ++it)
      {
        *it /= bck;
      }
    }
  }  
  
  struct Point { 
    // The column this point belongs to
    float x; 
    // The row this point belongs to
    float y; 
    // Data value (not used (yet?))
    float f;
    // 1st-order forward partial derivative in x
    float dfdx;
    // 2nd-order forward partial derivative in x
    float d2fdx;
  };

  struct CDF {
    CDF(const std::string &hdrFileName) {
      std::cout<<"cdf.h - constructing CDF..."<<std::endl;
      hdri.load(hdrFileName);

      // Build up luminance image
      std::vector<float> luminance(hdri.width * hdri.height);
    
      struct vec3 { float x, y, z; };
      struct vec4 { float x, y, z, w; };

      for (int y = 0; y < hdri.height; ++y)
      {
        for (int x = 0; x < hdri.width; ++x)
        {
          // That's not actually luminance, but might as well be..
          if (hdri.numComponents == 3)
          {
            vec3 rgb = *((vec3*)hdri.pixel.data() + y * hdri.width + x);
            luminance[y * hdri.width + x] = max(rgb.x,max(rgb.y,rgb.z));
          }
          else if (hdri.numComponents == 4)
          {
            vec4 rgba = *((vec4*)hdri.pixel.data() + y * hdri.width + x);
            luminance[y * hdri.width + x] = max(rgba.x,max(rgba.y,rgba.z));
          }
          else
            assert(0);
        }
      }
  
      // Build up CDF
      cumulatedRows.resize(hdri.width * hdri.height, 0);
      cumulatedLastCol.resize(hdri.height);
      std::vector<float> lastCol(hdri.height);

      for (int y = 0; y < hdri.height; ++y)
      {
        // Scan each row
        size_t off = y * hdri.width;
        scan(luminance.data() + off, luminance.data() + off + hdri.width, cumulatedRows.data() + off);
        // Assemble the last column by filling with the last item of each row
        lastCol[y] = *(cumulatedRows.data() + off + hdri.width - 1);

        // Normalize the row
        normalize(cumulatedRows.data() + off, cumulatedRows.data() + off + hdri.width);
      }

      // Scan and normalize the last column
      scan(lastCol.begin(), lastCol.end(), cumulatedLastCol.begin());
      normalize(cumulatedLastCol.begin(), cumulatedLastCol.end());
    }

    Mesh CDF::asTriangleMesh(Mesh::Representation repr,
                               float simplificationRate,
                               bool dumpAsObj) 
    {
      std::cout<<"cdf.h - converting to cdf geometry" << std::endl;
      unsigned maxControlPoints = hdri.height * hdri.width * (1.f - simplificationRate);

      Mesh res;
      std::vector<Point> controlPoints;

      // Sample the row CDFs, collecting first and second derivatives
      std::vector<Point> samples(hdri.width * hdri.height);
      for (int y=0; y<hdri.height; ++y) {
        float rowHeight = (y == 0) ? cumulatedLastCol[0] : cumulatedLastCol[y] - cumulatedLastCol[y-1];
        const float *row = cumulatedRows.data() + y*hdri.width;
        float prevSlope = 0.f;//row[1]-row[0];
        for (int x = 0; x < hdri.width - 1; ++x) {
          size_t index = y * hdri.width + x;
          float slope = row[x+1] - row[x];
          samples[index].x     = x + 1.f;
          samples[index].y     = y;
          samples[index].f     = row[x];
          samples[index].dfdx  = slope*rowHeight;
          samples[index].d2fdx = fabsf(slope-prevSlope)*rowHeight;
          prevSlope = slope;
        }
      }

      // Sort the samples by absolute differences in slope, in descending order
      std::sort(samples.begin(), samples.end(), [] (Point a, Point b) { return a.d2fdx > b.d2fdx; });

      // Accept the N most influential samples
      for (int i = 0; i < min((size_t)maxControlPoints,samples.size()); ++i) {
        if (samples[i].dfdx < FLT_MIN || samples[i].d2fdx < FLT_MIN) continue;
        controlPoints.push_back(samples[i]);
      }

      // Insert starts and stops for each row, guaranteeing each row is represented.
      for (int y=0; y<hdri.height; ++y) {
        controlPoints.push_back({0, float(y), 0.f, 0, 0});
        controlPoints.push_back({float(hdri.width), float(y), 1.f, 0, 0});
      }

      // Reorder the samples into a sparse, row major order image
      std::sort(controlPoints.begin(),controlPoints.end(), [](Point a, Point b) { return a.x < b.x; });
      std::stable_sort(controlPoints.begin(),controlPoints.end(), [](Point a, Point b) { return a.y < b.y; });

      // Count the number of control points used by each row
      std::vector<int> counts(hdri.height + 2, 0);
      for (unsigned i=0; i<controlPoints.size(); ++i) counts[controlPoints[i].y+1]++;
      
      // Take the prefix sum of these counts to compute addresses into each row
      for (unsigned i=1; i<counts.size(); ++i) counts[i] = counts[i-1]+counts[i];

      float prevRowCdf = 0.f;
      for (int y = 0; y < hdri.height; ++y)
      {
        // Merge any "empty" neighboring rows
        int rowCount = 1;
        int rowStart = y;
        while (y < (hdri.height - 1) && 
          ((counts[y+1] - counts[y+0]) == 2) && 
          ((counts[y+2] - counts[y+1]) == 2)) {
          rowCount++;
          y++;
        }

        float currRowCdf = cumulatedLastCol[y];
        Geometry geom;
        geom.tag = Geometry::CDF;
        geom.geomPdf = (currRowCdf - prevRowCdf) / float(rowCount); 
        geom.rowStart = rowStart;
        geom.rowCount = rowCount;

        // Generate triangle geometry to represent the CDF
        int o = 0;
        geom.vertex.push_back({prevRowCdf, 0.f, 0.f});
        geom.vertex.push_back({currRowCdf, 0.f, 0.f});
        for (unsigned i = counts[y] + 1; i != counts[y+1]; ++i) {
          // note, geometry dimensions span from 0-1 to improve traversal performance
          geom.vertex.push_back({prevRowCdf, controlPoints[i].f, float(controlPoints[i].x) / float(hdri.width)});
          geom.vertex.push_back({currRowCdf, controlPoints[i].f, float(controlPoints[i].x) / float(hdri.width)});
          geom.index.push_back({o + 0, o + 1, o + 3});
          geom.index.push_back({o + 3, o + 2, o + 0});
          geom.triPdfs.push_back((controlPoints[i].f - controlPoints[i-1].f) / float(controlPoints[i].x - controlPoints[i-1].x));
          geom.triPdfs.push_back((controlPoints[i].f - controlPoints[i-1].f) / float(controlPoints[i].x - controlPoints[i-1].x));
          o += 2;        
        }
        
        // Add the generated geometry to our list
        res.geoms.push_back(geom);
        prevRowCdf = currRowCdf;
      }
      return res;
    }                           

    // Containing *normalized* prefix sums of rows
    std::vector<float> cumulatedRows;

    // The last column, cumulated and normalized
    std::vector<float> cumulatedLastCol;

    // The original environment map
    HDRI hdri;

  }; 
}
