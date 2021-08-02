#pragma once

#include <memory>
#include <string>
#include <vector>
#include "owl/owl.h"
#include "owl/common/math/vec.h"

#include "tiny_obj_loader.h"

namespace cdf {

  using namespace owl;
  using namespace owl::common;
 
  struct Geometry {
    // Tag that helps describing the geometry a bit
    enum Tag {
      CDF,           // indicates that the geometry represents the CDF
      Model,         // indicates that the geometry belongs to a model
    };

    std::vector<vec3f> vertex;
    std::vector<vec3i> index;
    
    // for CDF types
    float geomPdf; 
    int rowStart;
    int rowCount;
    std::vector<float> triPdfs;

    Tag tag;
  };

  struct Mesh {
    enum Representation { Ribbons };

    std::vector<Geometry> geoms;

    Mesh() {}

    Mesh(std::string objFileName) {
      tinyobj::attrib_t attrib;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
      std::string err = "";

      std::string modelDir = objFileName.substr(0, objFileName.rfind('/') + 1);

      bool readOK
        = tinyobj::LoadObj(&attrib,
                          &shapes,
                          &materials,
                          &err,
                          &err,
                          objFileName.c_str(),
                          modelDir.c_str(),
                          /* triangulate */true,
                          /* default vertex colors fallback*/ false);

      if (!readOK)
        throw std::runtime_error("Could not read OBJ model from " + objFileName + " : " + err);

      for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
        tinyobj::shape_t& shape = shapes[shapeID];
        Geometry geom;
        geom.tag = Geometry::Model;
        // Just lazily copy _all_ the vertices into this geom..
        for (std::size_t i=0; i<attrib.vertices.size(); i+=3) {
          float *v = attrib.vertices.data() + i;
          geom.vertex.push_back({v[0],v[1],v[2]});
        }
        for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
          if (shape.mesh.num_face_vertices[faceID] != 3)
            throw std::runtime_error("not properly tessellated"); // while this would actually be rather easy...
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          geom.index.push_back({idx0.vertex_index,
                                idx1.vertex_index,
                                idx2.vertex_index});
        }
        geoms.push_back(geom);
      }
    }
  };

}
